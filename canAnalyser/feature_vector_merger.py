import pandas as pd
from sklearn.impute import KNNImputer


class FeatureVectorMerger:
    @staticmethod
    def extract_common_base(filename):
        parts = filename.rsplit('_', 2)
        if len(parts) > 2:
            return '_'.join(parts[:-1])
        return filename


    def merge_feature_vectors(self, *feature_vectors):
        """
        Merges multiple feature vectors based on a common base extracted from the indices.

        Returns:
        DataFrame: Merged feature vector.
        """
        def imputer(data):
            # Impute remaining NaN values
            data =  data.dropna(axis=1, how='all')
            imputer = KNNImputer(n_neighbors=5, weights="uniform")
            imputed_data = imputer.fit_transform(data)
            return pd.DataFrame(imputed_data, columns=data.columns, index=data.index)

        merged_fv = None
        for fv in feature_vectors:
            fv.index = fv.index.map(self.extract_common_base)
            if merged_fv is None:
                merged_fv = fv
            else:
                merged_fv = merged_fv.merge(fv, left_index=True, right_index=True, how='inner')
        
        try:
            if "error_code_mean" in merged_fv.columns:
                error_codes = merged_fv["error_code_mean"]
                merged_fv = merged_fv.drop(["error_code_mean"], axis=1)
            if "error_code_median" in merged_fv.columns:
            	error_codes = merged_fv["error_code_median"]
            	merged_fv = merged_fv.drop(["error_code_median"], axis=1)
            if "error_code_kurtosis" in merged_fv.columns:
            	error_codes = merged_fv["error_code_kurtosis"]
            	merged_fv = merged_fv.drop(["error_code_kurtosis"], axis=1)
            if "error_code" in merged_fv.columns:
                error_codes = merged_fv["error_code"]
                merged_fv = merged_fv.drop(["error_code"], axis=1)
            
            if "rt_ccu_evstatus_errorcode" in merged_fv.columns:
                merged_fv = merged_fv.drop(["rt_ccu_evstatus_errorcode"], axis=1)
            elif "rt-ccu_evstatus_errorcode" in merged_fv.columns:
                merged_fv = merged_fv.drop(["rt-ccu_evstatus_errorcode"], axis=1)

        except Exception as e:
            print(f"Error during dropping error codes: {e}")
        imputed_fv = imputer(merged_fv)

        return imputed_fv, error_codes

## Example usage:
## Assuming 'ccu_fv', 'iso_fv', 'ts_fv', 'wsgt_fv', 'q7_fv', 'fp_fv' are your individual feature vectors
#merger = FeatureVectorMerger()
#fv = merger.merge_feature_vectors(ccu_fv, iso_fv, ts_fv, wsgt_fv, q7_fv, fp_fv)

