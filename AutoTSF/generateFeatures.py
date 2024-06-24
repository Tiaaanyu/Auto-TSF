#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 23:05
# @Author  : ZSH
# from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.transformations.panel.dictionary_based import PAA,SFA,SAX
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
from sktime.transformations.panel import dwt
from sktime.transformations.series import acf
import pickle
from featureUtil import *
import numpy as np
import tsfresh as tsf
import pandas as pd
from pymfe.mfe import MFE
# from load import convert_tsf_to_dataframe
import math
from tsai.basics import *

class GenerateFeature():
    
    def __init__(self,name):
        "ETTh1"
        min_ = 10000
        print("processing: " + name)
        if any(dname in name for dname in ['m4', 'nn5', 'tourism', 'electricity']):
            self.X=get_Monash_forecasting_data(name)
            ts = pd.unique(self.X.series_name)
            print("ts num: " + str(len(ts)))
            series = []
            for t in ts:
                series.append(pd.Series(self.X[self.X['series_name'] == t]['series_value']))
            dataset = dict()
            min_len = min([len(s) for s in series] + [min_])
            for idx, t in enumerate(ts):
                dataset[t] = {'series_value':series[idx][:min_len]}

        if "ETT" in name:
            self.X=get_long_term_forecasting_data(name)
            names = self.X.columns[1:]
            n_patch = 30
            x_ = np.array_split(self.X, n_patch)
            min_len = min([len(x) for x in x_] + [min_])
            dataset = {f"T{idx}":{n: pd.Series(x[n])[:min_len] for n in names} for idx, x in enumerate(x_)}
            print("ts num: " + str(len(x_)))

        
        self.X = pd.DataFrame(dataset)
        self.X = self.X.T
        self.X_nd=from_nested_to_3d_numpy(self.X)
        self.time_feature_num=26
        
    def _generate_Time_domain_features(self):
 
        n_paa = 1
        n_sax_symbols = 2
        n_sax_symbols_slope = 4
        n_sax_symbols_avg = 4
        ##PAA
        paa = PAA(n_paa)
        features_paa = paa.fit_transform(self.X)  
        ##SAX,1d-SAX
        sax = SymbolicAggregateApproximation(n_segments=n_paa, alphabet_size_avg=n_sax_symbols)
        one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa, alphabet_size_avg=n_sax_symbols_avg,
                                                        alphabet_size_slope=n_sax_symbols_slope)

        X_nd = np.array(self.X_nd)
        X_nd = X_nd.reshape((X_nd.shape[0], 1, -1))
        features_SAX = sax.inverse_transform(sax.fit_transform(X_nd))
        features_1d_SAX = one_d_sax.inverse_transform(one_d_sax.fit_transform(X_nd))
        features_paa = from_nested_to_3d_numpy(features_paa)
        features_SAX = np.array(features_SAX).reshape((X_nd.shape[0], X_nd.shape[1], -1))
        features_1d_SAX = np.array(features_1d_SAX).reshape((X_nd.shape[0], X_nd.shape[1], -1))
        return features_paa, features_SAX, features_1d_SAX

    def _generate_Frequency_domain_features(self):
       
        ##SFA
        X_np = self.X_nd
        X_np = np.array(X_np)
        X_np = X_np.reshape((X_np.shape[0], 1, -1))
        sfa = SFA()
        features_sfa = sfa.fit_transform(X_np)
        features_sfa = np.array(features_sfa).reshape((X_np.shape[0], X_np.shape[1], -1))
        ##DWT
        Dwt = dwt.DWTTransformer()
        features_dwt = Dwt.fit_transform(self.X)
        features_dwt = from_nested_to_3d_numpy(features_dwt)
        return features_sfa, features_dwt

    def _generate_Evolution_features(self):
       
        ##ACF,PACF
        Acf = acf.AutoCorrelationTransformer()
        X_np = self.X_nd
        features_ACF = []
        features_PACF = []
        pacf=PACF()
        for i in range(X_np.shape[0]):
            temp_res = []
            temp_res_pacf = []
            for j in range(X_np.shape[1]):
                try:
                    acf_ = (Acf.fit_transform(X_np[i][j]))
                    pacf_ = (pacf.cal_my_pacf_yw(X_np[i][j]))
                except:
                    pass
                temp_res.append(acf_)
                temp_res_pacf.append(pacf_)
            features_ACF.append(temp_res)
            features_PACF.append(temp_res_pacf)
        features_ACF = np.array(features_ACF)
        features_PACF = np.array(features_PACF)
        return features_ACF, features_PACF

    def _generate_tsfresh_features(self):
      
        X = self.X_nd
        features_abs_energy = []
        features_absolute_sum_of_changes = []
        features_approximate_entropy = []
        features_autocorrelation = []
        features_binned_entropy = []
        features_c3 = []
        features_cid_ce = []
        features_count_above_mean = []
        features_cwt_coefficients = []
        features_fft_aggregated = []
        features_energy_ratio_by_chunks = []
        features_kurtosis = []
        features_longest_strike_above_mean = []
        features_mean_abs_change = []
        features_mean_second_derivative_central = []
        features_number_cwt_peaks = []
        features_percentage_of_reoccurring_datapoints_to_all_datapoints = []
        features_root_mean_square = []
        features_sample_entropy = []
        features_skewness = []

        for i in range(X.shape[0]):
            abs_energy = []
            absolute_sum_of_changes = []
            approximate_entropy = []
            autocorrelation = []
            binned_entropy = []
            c3 = []
            cid_ce = []
            count_above_mean = []
            cwt_coefficients = []
            fft_aggregated = []
            energy_ratio_by_chunks = []
            kurtosis = []
            longest_strike_above_mean = []
            mean_abs_change = []
            mean_second_derivative_central = []
            number_cwt_peaks = []
            percentage_of_reoccurring_datapoints_to_all_datapoints = []
            root_mean_square = []
            sample_entropy = []
            skewness = []

            for j in range(X.shape[1]):
                ts = pd.Series(X[i][j])
                abs_energy.append(tsf.feature_extraction.feature_calculators.abs_energy(ts))
                absolute_sum_of_changes.append(tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(ts))
                approximate_entropy.append(tsf.feature_extraction.feature_calculators.approximate_entropy(ts, 10, 0.1))
                autocorrelation.append(tsf.feature_extraction.feature_calculators.autocorrelation(ts, 2))
                binned_entropy.append(tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10))
                c3.append(tsf.feature_extraction.feature_calculators.c3(ts, 2))
                cid_ce.append(tsf.feature_extraction.feature_calculators.cid_ce(ts, True))
                count_above_mean.append(tsf.feature_extraction.feature_calculators.count_above_mean(ts))

                param_cwt = [{'widths': tuple([2, 2, 2]), 'coeff': 2, 'w': 2}]
                cwt_coefficients.append(
                    list(tsf.feature_extraction.feature_calculators.cwt_coefficients(ts, param_cwt))[0][1])

                param_fft_aggregated = [{'aggtype': 'skew'}]
                fft_aggregated.append(
                    list(tsf.feature_extraction.feature_calculators.fft_aggregated(ts, param_fft_aggregated))[0][1])

                param_ratio_by_chunks = [{'num_segments': 10, 'segment_focus': 5}]
                energy_ratio_by_chunks.append(
                    list(tsf.feature_extraction.feature_calculators.energy_ratio_by_chunks(ts, param_ratio_by_chunks))[
                        0][1])

                kurtosis.append(tsf.feature_extraction.feature_calculators.kurtosis(ts))
                longest_strike_above_mean.append(
                    tsf.feature_extraction.feature_calculators.longest_strike_above_mean(ts))
                mean_abs_change.append(tsf.feature_extraction.feature_calculators.mean_abs_change(ts))
                mean_second_derivative_central.append(
                    tsf.feature_extraction.feature_calculators.mean_second_derivative_central(ts))
                number_cwt_peaks.append(tsf.feature_extraction.feature_calculators.number_cwt_peaks(ts, 10))
                percentage_of_reoccurring_datapoints_to_all_datapoints.append(
                    tsf.feature_extraction.feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(
                        ts))
                root_mean_square.append(tsf.feature_extraction.feature_calculators.sample_entropy(ts))
                sample_entropy.append(tsf.feature_extraction.feature_calculators.sample_entropy(ts))
                skewness.append(tsf.feature_extraction.feature_calculators.skewness(ts))

            features_abs_energy.append(abs_energy)
            features_absolute_sum_of_changes.append(absolute_sum_of_changes)
            features_approximate_entropy.append(approximate_entropy)
            features_autocorrelation.append(autocorrelation)
            features_binned_entropy.append(binned_entropy)
            features_c3.append(c3)
            features_cid_ce.append(cid_ce)
            features_count_above_mean.append(count_above_mean)
            features_cwt_coefficients.append(cwt_coefficients)
            features_fft_aggregated.append(fft_aggregated)
            features_energy_ratio_by_chunks.append(energy_ratio_by_chunks)
            features_kurtosis.append(kurtosis)
            features_longest_strike_above_mean.append(longest_strike_above_mean)
            features_mean_abs_change.append(mean_abs_change)
            features_mean_second_derivative_central.append(mean_second_derivative_central)
            features_number_cwt_peaks.append(number_cwt_peaks)
            features_percentage_of_reoccurring_datapoints_to_all_datapoints.append(
                percentage_of_reoccurring_datapoints_to_all_datapoints)
            features_root_mean_square.append(root_mean_square)
            features_sample_entropy.append(sample_entropy)
            features_skewness.append(skewness)
        features_abs_energy = np.array(features_abs_energy)
        features_absolute_sum_of_changes = np.array(features_absolute_sum_of_changes)
        features_approximate_entropy = np.array(features_approximate_entropy)
        features_autocorrelation = np.array(features_autocorrelation)
        features_binned_entropy = np.array(features_binned_entropy)
        features_c3 = np.array(features_c3)
        features_cid_ce = np.array(features_cid_ce)
        features_count_above_mean = np.array(features_count_above_mean)
        features_cwt_coefficients = np.array(features_cwt_coefficients)
        features_fft_aggregated = np.array(features_fft_aggregated)
        features_energy_ratio_by_chunks = np.array(features_energy_ratio_by_chunks)
        features_kurtosis = np.array(features_kurtosis)
        features_longest_strike_above_mean = np.array(features_longest_strike_above_mean)
        features_mean_abs_change = np.array(features_mean_abs_change)
        features_mean_second_derivative_central = np.array(features_mean_second_derivative_central)
        features_number_cwt_peaks = np.array(features_number_cwt_peaks)
        features_percentage_of_reoccurring_datapoints_to_all_datapoints = np.array(
            features_percentage_of_reoccurring_datapoints_to_all_datapoints)
        features_root_mean_square = np.array(features_root_mean_square)
        features_sample_entropy = np.array(features_sample_entropy)
        features_skewness = np.array(features_skewness)
        return features_abs_energy, features_absolute_sum_of_changes, features_approximate_entropy, features_autocorrelation, features_binned_entropy, \
               features_c3, features_cid_ce, features_count_above_mean, features_cwt_coefficients, features_fft_aggregated, features_energy_ratio_by_chunks, \
               features_kurtosis, features_longest_strike_above_mean, features_mean_abs_change, features_mean_second_derivative_central, features_number_cwt_peaks, \
               features_percentage_of_reoccurring_datapoints_to_all_datapoints, features_root_mean_square, features_sample_entropy, features_skewness

    def generate_time_features(self):
        features_paa, features_SAX, features_1d_SAX = self._generate_Time_domain_features()
        features_sfa, features_dwt = self._generate_Frequency_domain_features()
        features_ACF, features_PACF = self._generate_Evolution_features()
        features_abs_energy, features_absolute_sum_of_changes, features_approximate_entropy, features_autocorrelation, features_binned_entropy, \
        features_c3, features_cid_ce, features_count_above_mean, features_cwt_coefficients, features_fft_aggregated, features_energy_ratio_by_chunks, \
        features_kurtosis, features_longest_strike_above_mean, features_mean_abs_change, features_mean_second_derivative_central, features_number_cwt_peaks, \
        features_percentage_of_reoccurring_datapoints_to_all_datapoints, features_root_mean_square, features_sample_entropy, features_skewness = self._generate_tsfresh_features()
        self.features = {
            'features_paa': features_paa,
            'features_SAX': features_SAX,
            'features_1d_SAX': features_1d_SAX,
            'features_dwt': features_dwt,
            'features_ACF': features_ACF,
            'features_PACF': features_PACF,
            'features_abs_energy': features_abs_energy,
            'features_absolute_sum_of_changes': features_absolute_sum_of_changes,
            'features_approximate_entropy': features_approximate_entropy,
            'features_autocorrelation': features_autocorrelation,
            'features_binned_entropy': features_binned_entropy,
            'features_c3': features_c3,
            'features_cid_ce': features_cid_ce,
            'features_count_above_mean': features_count_above_mean,
            'features_cwt_coefficients': features_cwt_coefficients,
            'features_fft_aggregated': features_fft_aggregated,
            'features_energy_ratio_by_chunks': features_energy_ratio_by_chunks,
            'features_kurtosis': features_kurtosis,
            'features_longest_strike_above_mean': features_longest_strike_above_mean,
            'features_mean_abs_change': features_mean_abs_change,
            'features_mean_second_derivative_central': features_mean_second_derivative_central,
            'features_number_cwt_peaks': features_number_cwt_peaks,
            'features_percentage_of_reoccurring_datapoints_to_all_datapoints': features_percentage_of_reoccurring_datapoints_to_all_datapoints,
            'features_root_mean_square': features_root_mean_square,
            'features_sample_entropy': features_sample_entropy,
            'features_skewness': features_skewness
        }
        print("time features num: " + str(len(self.features)))
        return self.features
    def _time_feature_transform(self):
        features_dic=self.features
        size = features_dic['features_paa'].shape[0]

        self.time_features = np.zeros((size, self.time_feature_num))
        i = 0
        for key in features_dic.keys():
            
            self.time_features[:, i] = np.sum(features_dic[key].reshape((size, -1)), axis=1)
            i += 1
        return self.time_features
    def _generate_meata_features(self):
        features_nd = self.time_features
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        mfe.fit(features_nd)
        ft = mfe.extract()
        self.meta_features = {}
        meta_features_list = ['attr_to_inst', 'freq_class.mean', 'inst_to_attr', 'nr_attr', 'can_cor.mean', 'cor.mean',
                              'iq_range.mean', 'kurtosis.mean', 'lh_trace', 'mad.mean', 'median.mean', 'nr_disc',
                              'p_trace',
                              'roy_root', 'skewness.mean', 't_mean.mean', 'w_lambda', 'attr_conc.mean', 'attr_ent.mean',
                              'class_conc.mean', 'class_ent', 'class_ent', 'joint_ent.mean', 'mut_inf.mean', 'ns_ratio']
        for i in range(len(ft[0])):
            if ft[0][i] in meta_features_list:
                self.meta_features[ft[0][i]] = (0 if math.isinf(ft[1][i])or math.isnan(ft[1][i]) else ft[1][i])
        return self.meta_features
    
    def generate_meta_features(self):
        self.generate_time_features()
        self._time_feature_transform()
        self._generate_meata_features()
        self.meta_feature_list=[] 
        for key in self.meta_features.keys():
            self.meta_feature_list.append(self.meta_features[key])
        print("time features shape: " + str(self.time_features.shape))
        return list(np.mean(self.time_features, axis=0)) + self.meta_feature_list

def get_meta_data(data_names: str) -> dict:
    meta_data = dict()
    for d in data_names:
        ge = GenerateFeature(d)
        meta_data[d] = ge.generate_meta_features()
    for meta in meta_data:
        meta_ = np.array(meta_data[meta])
        meta_ = np.nan_to_num(meta_)
        # largest_float32 = np.finfo(np.float32).max
        # meta_ = np.where(np.isinf(meta_), 1e7, meta_)
        max_value = np.finfo(np.float32).max
        min_value = np.finfo(np.float32).min
        meta_ = np.clip(meta_, min_value, max_value)
        meta_data[meta] = meta_
    return meta_data

if __name__=="__main__":
    # path='../data/BME/BME_TRAIN.ts'
    datas = [
    # "ETTh1", # 210.40382504463196
    # "ETTh2", # 1247.633891582489
    # "ETTm1", # 986.3248279094696
    "ETTm2",
    # 'm4_yearly_dataset',
    'm4_quarterly_dataset',
    'm4_monthly_dataset',
    'm4_weekly_dataset',
    'm4_daily_dataset',
    # 'm4_hourly_dataset',
    # "nn5_weekly_dataset",
    "nn5_daily_dataset_without_missing_values",
    'electricity_hourly_dataset', 
    'electricity_weekly_dataset',
    'tourism_yearly_dataset',
    'tourism_quarterly_dataset',
    'tourism_monthly_dataset'
    ]
    import time
    with open("meta_feature_2.txt", mode='w') as f:
        f.write("meta_data = dict()\n")
        for ds in datas:
            ge=GenerateFeature(ds)
            prev_time = time.time()
            f.write(f"meta_data['{ds}']=" + str(ge.generate_meta_features())+"\n")
            cur_time = time.time() - prev_time
            print("Time: " + str(cur_time))
            f.flush()

