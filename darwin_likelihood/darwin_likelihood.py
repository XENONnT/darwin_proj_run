import numpy as np
from scipy.interpolate import interp1d
import multihist as mh
from inference_interface import template_to_multihist
import blueice as bi
import warnings 
from distutils.version import LooseVersion
from itertools import product
import json
import wimprates as wr
import scipy.stats as sps
from copy import deepcopy


from binference.simulators import simulate_interpolated
from binference.template_source import TemplateSource
#from binference.utils import read_neyman_threshold


from blueice.likelihood import UnbinnedLogLikelihood, LogAncillaryLikelihood, LogLikelihoodSum
from blueice.inference import bestfit_scipy, one_parameter_interval

from inference_interface import toydata_to_file, toydata_from_file, structured_array_to_dict, dict_to_structured_array
from os import path
import pkg_resources



can_check_binning = True
minimize_kwargs = {'method': 'Powell', 'options': {'maxiter': 10000000}}
science_slice_args = [{'slice_axis': 0, 'sum_axis': False, 'slice_axis_limits': [3.01, 99.9]}]

def get_spectrum(fn):
    """
        Translates bbf-style JSON files to spectra. 
        units are keV and /kev*day*kg
    """
    contents = json.load(open(fn,"r"))
    print(contents["description"])
    esyst = contents["coordinate_system"][0][1]
    ret = interp1d(np.linspace(*esyst), contents["map"],bounds_error=False,fill_value=0.)
    return ret

class SpectrumTemplateSource(bi.HistogramPdfSource):
    """
        configuration parameters: 
        spectrum_name: name of bbf json-like spectrum _OR_ function that can be called 
        templatename #3D histogram (Etrue,S1,S2) to open
        histname: histogram name
        named_parameters: list of config settings to pass to .format on histname and filename 
    """
    def build_histogram(self):
        format_dict = {k: self.config[k] for k in self.config.get('named_parameters', [])}
        templatename = self.config['templatename'].format(**format_dict)
        histname = self.config['histname'].format(**format_dict)

        spectrum = self.config["spectrum"]
        if type(spectrum) is str:
            spectrum = get_spectrum(spectrum.format(**format_dict))

        slice_args = self.config.get("slice_args",{})
        if type(slice_args) is dict:
            slice_args = [slice_args]


        h = template_to_multihist(templatename, histname)
        
        #Perform E-scaling
        ebins = h.bin_edges[0]
        ecenters = 0.5 * (ebins[1::] + ebins[0:-1])
        ediffs =         (ebins[1::] - ebins[0:-1])
        h.histogram = h.histogram * (spectrum(ecenters)*ediffs)[:,None,None]
        h = h.sum(axis=0) #remove energy-axis


        for sa in slice_args:
            slice_axis = sa.get("slice_axis",None)
            sum_axis = sa.get("sum_axis",False) #decide if you wish to sum the histogram into lower dimensions or 
            
            slice_axis_limits = sa.get("slice_axis_limits",[0,0])
            collapse_axis = sa.get('collapse_axis', None)
            collapse_slices = sa.get('collapse_slices', None)
        


            if (slice_axis is not None) :
                if sum_axis:
                    h = h.slicesum(axis=slice_axis,
                                   start = slice_axis_limits[0],
                                   stop=slice_axis_limits[1])
                else:
                    h = h.slice(axis=slice_axis,
                                start = slice_axis_limits[0],
                                stop=slice_axis_limits[1])

            if collapse_axis is not None:
                if collapse_slices is None:
                    raise ValueError("To collapse you must supply collapse_slices")
                h = h.collapse_bins(collapse_slices, axis=collapse_axis)

        self.dtype = []
        for n,_ in self.config['analysis_space']:
            self.dtype.append((n,float))
        self.dtype.append(('source',int))

        
        # Fix the bin sizes
        if can_check_binning:
            # Deal with people who have log10'd their bins
            for axis_i in self.config.get('log10_bins', []):
                h.bin_edges[axis_i] = 10**h.bin_edges[axis_i]

            # Check if the histogram bin edges are correct
            for axis_i, (_, expected_bin_edges) in enumerate(self.config['analysis_space']):
                expected_bin_edges = np.array(expected_bin_edges)            
                seen_bin_edges = h.bin_edges[axis_i]
                if len(seen_bin_edges) != len(expected_bin_edges):
                    raise ValueError("Axis %d of histogram %s in root file %s has %d bin edges, but expected %d" % (
                        axis_i, histname, templatename, len(seen_bin_edges), len(expected_bin_edges)
                    ))
                try:
                    np.testing.assert_almost_equal(seen_bin_edges / expected_bin_edges, 
                                                   np.ones_like(seen_bin_edges), decimal=4)
                except AssertionError:
                    warnings.warn("Axis %d of histogram %s in root file %s has bin edges %s, "
                                  "but expected %s. Since length matches, setting it expected values..." % (
                                      axis_i, histname, templatename, seen_bin_edges, expected_bin_edges
                                  ))
                    h.bin_edges[axis_i] = expected_bin_edges

        self._bin_volumes = h.bin_volumes()      # TODO: make alias
        self._n_events_histogram = h.similar_blank_histogram()    # Shouldn't be in HistogramSource... anyway

        if self.config.get('normalise_template',False):
            h /= h.n

        
        h *= self.config.get('histogram_multiplier', 1)
           
        # Convert h to density...
        if self.config.get('in_events_per_bin'):
            h.histogram /= h.bin_volumes()
        self.events_per_day = (h.histogram * self._bin_volumes).sum()

        
        # ... and finally to probability density
        h.histogram /= self.events_per_day
        self._pdf_histogram = h
    def simulate(self, n_events):
        dtype = []
        for n,_ in self.config['analysis_space']:
            dtype.append((n,float))
        dtype.append(('source',int))
        ret = np.zeros(n_events, dtype=dtype)
        #t = self._pdf_histogram.get_random(n_events)
        h = self._pdf_histogram * self._bin_volumes
        t = h.get_random(n_events)
        for i,(n,_) in enumerate(self.config.get('analysis_space')):
            ret[n] = t[:,i]
        return ret 

def get_resource_filename(fname=""):
    return pkg_resources.resource_filename("darwin_likelihood","data/"+fname)


def get_likelihood_config(exposure=1.,signal_config={},**kwargs):
    source_configs = [
            dict(
                name="ernusun",
                label="ER events from B8",
                templatename = get_resource_filename("NEST_dummyresponse.hdf5"),
                histname="er",
                spectrum=get_resource_filename("spectra/SolarNeutrinoFEASpectrum.json"),
            ),
            dict(
                name="nrnusun",
                label="CEvNS events from B8 and HEP",
                templatename = get_resource_filename("NEST_dummyresponse.hdf5"),
                histname="nr",
                spectrum=get_resource_filename("spectra/Solar_CNNSSpectrum.json"),
            ),
            dict(
                name="xe136",
                label="Xe136",
                templatename = get_resource_filename("NEST_dummyresponse.hdf5"),
                histname="er",
                spectrum=get_resource_filename("spectra/Xe136Spectrum.json"),
            ),
            dict(
                name="atmnu",
                label="Atmospheric Neutrinos",
                templatename = get_resource_filename("NEST_dummyresponse.hdf5"),
                histname="nr",
                spectrum=get_resource_filename("spectra/Atm_CNNSSpectrum.json"),
            ),
            dict(
                name="snnu",
                label="(Diffuse) Supernova Neutrinos",
                templatename = get_resource_filename("NEST_dummyresponse.hdf5"),
                histname="nr",
                spectrum=get_resource_filename("spectra/DSN_CNNSSpectrum.json"),
            ),
            signal_config,
            ]
    ll_config = dict(
        analysis_space=[("cs1",np.linspace(3,100,100-3+1)),
                        ("cs2",np.logspace(np.log10(200),np.log10(10000),101))], 
            slice_args = science_slice_args,
            sources = source_configs,
            default_source_class=SpectrumTemplateSource, 
            livetime_days=exposure*365.*1000.,
            in_events_per_bin=True,
            log10_bins=[],
            )
    return ll_config


parameter_uncerts = dict(
        ernusun = 0.02,
        nrnusun = 0.02,
        xe136 = 0.1,
        atmnu = 0.1,
        snnu = 0.1,
        )

class InferenceObject:
    def __init__(self, wimp_mass = 50, livetime = 1.,
            ll_config_overrides={},
            limit_threshold = lambda x,dummy:0.5*sps.chi2(1).isf(0.1),
            **kwargs):
        
        signal_spectrum = lambda x: wr.rate_wimp_std(x,mw=wimp_mass,sigma_nucleon=1e-45) / (365.*1000.)
        signal_config = dict(
            histname= 'nr',
            label= 'WIMP events',
            name= 'signal',
            spectrum=signal_spectrum,
            templatename= '/Users/kdund/Desktop/Darwin_projection/darwin_proj_run/darwin_likelihood/data/NEST_dummyresponse.hdf5',        never_save_to_cache=True,
            dont_hash_settings = ["spectrum"],
        )
        self.limit_threshold = limit_threshold
        ll_config = get_likelihood_config(exposure = livetime, signal_config=signal_config)
        ll_config.update(ll_config_overrides)
        ll_config["livetime_days"] = livetime
        ll = UnbinnedLogLikelihood(ll_config)

        for parameter in ["ernusun","nrnusun","xe136","atmnu","snnu"]:
            ll.add_rate_parameter(parameter,
                    log_prior = sps.norm(1.,parameter_uncerts[parameter]).logpdf)

        ll.add_rate_parameter("signal")

        ll.prepare()

        dtype = [("cs1",float),("cs2",float)]
        dummy_data = np.zeros(1,dtype)
        dummy_data["cs1"] = 50.
        dummy_data["cs2"] = 1000
        ll.set_data(dummy_data)

        self.lls = [ll]
        self.rgs =[simulate_interpolated(ll) for ll in self.lls]
        self.ll = LogLikelihoodSum(self.lls)

    def simulate_and_assign_data(self, generate_args = {}):
        datas = [rg.simulate(**generate_args) for rg in self.rgs]
        for data, ll in zip(datas, self.lls):
            ll.set_data(data)

    def simulate_and_assign_measurements(self,generate_args={}):

        for parameter_name in ["ernusun","nrnusun","xe136","atmnu","snnu"]:
            parameter_uncert = generate_args.get(parameter_name+"_uncert",parameter_uncerts[parameter_name])
            parameter_mean = generate_args.get(parameter_name+"_rate_multiplier",1)
            parameter_meas = max(0,sps.norm(parameter_mean, parameter_uncert).rvs())
            self.lls[0].rate_parameters[parameter_name]=sps.norm(parameter_meas,parameter_uncert).logpdf

    def llr(self, extra_args={}, extra_args_null={"signal_rate_multiplier":0.},guess={}):
        extra_args_null_total = deepcopy(extra_args)
        extra_args_null_total.update(extra_args_null)
        res1, llval1 = bestfit_scipy(self.ll, guess=guess,minimize_kwargs=minimize_kwargs, **extra_args)
        res0, llval0 = bestfit_scipy(self.ll, guess=guess,minimize_kwargs=minimize_kwargs,**extra_args_null_total)
        return 2.* (llval1-llval0), llval1, res1, llval0, res0
    def confidence_interval(self,llval_best,extra_args={},guess={},parameter_name = "signal_rate_multiplier",two_sided = True):

        #the confidence interval computation looks in a bounded region-- we will say that we will not look for larger than 300 signal events 
        rate_multiplier_max = 10000. / self.get_mus().get( parameter_name.replace("_rate_multiplier",""),1.)
        rate_multiplier_min = 0.

        dl = -1*np.inf
        ul = one_parameter_interval(self.ll, parameter_name,
                rate_multiplier_max,bestfit_routine = bestfit_scipy,
                minimize_kwargs = minimize_kwargs,
                t_ppf=self.limit_threshold, 
                guess = guess,**extra_args)
        if two_sided: 
            extra_args_null = deepcopy(extra_args)
            extra_args_null[parameter_name] = rate_multiplier_min

            res_null, llval_null = bestfit_scipy(self.ll, guess=guess, minimize_kwargs=minimize_kwargs, **extra_args_null)
            llr =  2.*(llval_best - llval_null)
            if llr <= self.limit_threshold(rate_multiplier_min,0):
                dl = rate_multiplier_min
            else:
                dl = one_parameter_interval(self.ll, parameter_name,
                    rate_multiplier_min,
                    kind = "lower",
                    bestfit_routine = bestfit_scipy,
                    minimize_kwargs = minimize_kwargs,
                    t_ppf=self.limit_threshold, 
                    guess = guess,**extra_args)
        return dl, ul


    def toy_simulation(self,generate_args={}, extra_args=[{},{"signal_rate_multiplier":0.}], guess={},compute_confidence_interval = False, confidence_interval_args = {}):
        self.simulate_and_assign_data(generate_args=generate_args)
        self.simulate_and_assign_measurements(generate_args=generate_args)
        self.ll = LogLikelihoodSum(self.lls)
        ress = []
        extra_args_runs = extra_args
        if type(extra_args_runs) is dict: 
            extra_args_runs = [extra_args_run]
        for extra_args_run in extra_args_runs:
            res, llval = bestfit_scipy(self.ll, guess=guess, minimize_kwargs=minimize_kwargs, **extra_args_run)
            res.update(extra_args_run)
            res["ll"] = llval
            res["dl"] = -1.
            res["ul"] = -1.
            ress.append(res)

        if compute_confidence_interval:
            ci_guess = deepcopy(ress[0])
            ci_guess.pop("signal_rate_multiplier",None)
            ci_args = {"llval_best":ress[0]["ll"], "extra_args":extra_args_runs[0],"guess":ci_guess}
            ci_args.update(confidence_interval_args)
            dl, ul = self.confidence_interval(**ci_args)
            ress[0]["dl"] = dl
            ress[0]["ul"] = ul



        return ress
    def get_mus(self,**res):
        ret = {}
        for ll in self.lls:
            mus = ll(full_output=True,**res)[1]
            for n,mu in zip(ll.source_name_list, mus):
                ret[n] = ret.get(n,0)+mu
        return ret
    def get_parameter_list(self):
        ret = [n + "_rate_multiplier" for n in list(self.ll.rate_parameters.keys())]
        ret += list(self.ll.shape_parameters.keys())
        ret +=["ll", "dl",  "ul"]
        return ret




        

