from sklearn.base import TransformerMixin
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

class _CustomTransformer(TransformerMixin):
	''' Data transformer class which validates data shapes. 
		Child classes should override _fit, _transform, _inverse_transform '''
	_input_shape  = None 
	_output_shape = None

	def fit(self, X, *args, **kwargs):				 
		self._input_shape = X.shape[1]
		self._fit(X.copy(), *args, **kwargs)
		return self 

	def transform(self, X, *args, **kwargs):
		self._validate_shape(X, self._input_shape)
		X = self._transform(X.copy(), *args, **kwargs)
		self._validate_shape(X, self._output_shape)
		self._output_shape = X.shape[1]
		return X 

	def inverse_transform(self, X, *args, **kwargs):
		self._validate_shape(X, self._output_shape)
		X = self._inverse_transform(X.copy(), *args, **kwargs)
		self._validate_shape(X, self._input_shape)
		self._input_shape = X.shape[1]
		return X 

	@staticmethod
	def config_info(*args, **kwargs):                 return '' # Return any additional info to construct model config
	def _fit(self, X, *args, **kwargs):               pass
	def _transform(self, X, *args, **kwargs):         raise NotImplemented
	def _inverse_transform(self, X, *args, **kwargs): raise NotImplemented
	def _validate_shape(self, X, shape):              assert(shape is None or X.shape[1] == shape), \
		f'Number of data features changed: expected {shape}, found {X.shape[1]}'
class LogTransformer(_CustomTransformer):
    ''' Transform into log domain '''

    def _transform(self, X, *args, **kwargs):
        return np.log1p(X)  # log1p to handle log(0)

    def _inverse_transform(self, X, *args, **kwargs):
        return np.expm1(X)  # expm1 to reverse log1p


class TransformerPipeline(_CustomTransformer):
    ''' Apply multiple transformers seamlessly '''

    def __init__(self, scalers=[]):
        self.scalers = scalers

    def _fit(self, X, *args, **kwargs):
        for scaler in self.scalers:
            X = scaler.fit_transform(X, *args, **kwargs)
        return self

    def _transform(self, X, *args, **kwargs):
        for scaler in self.scalers:
            X = scaler.transform(X, *args, **kwargs)
        return X

    def _inverse_transform(self, X, *args, **kwargs):
        for scaler in self.scalers[::-1]:
            X = scaler.inverse_transform(X, *args, **kwargs)
        return X

    def fit_transform(self, X, *args, **kwargs):
        # Manually apply a fit_transform to avoid transforming twice
        for scaler in self.scalers:
            X = scaler.fit_transform(X, *args, **kwargs)
        return X
def serialize(scaler, args=[], kwargs={}):
    return (scaler, args, kwargs)


class Preprocessor:
    def __init__(self,
                 scaler_type='robust',
                 scaler_scope='wavelength',
                 keep_shape=False,
                 keep_1st_derivative=False,
                 normalize_1st_derivative=False,
                 keep_2nd_derivative=False,
                 normalize_2nd_derivative=False):
        self.scaler_type = scaler_type
        self.scaler_scope = scaler_scope
        self.keep_shape = keep_shape
        self.keep_1st_derivative = keep_1st_derivative
        self.normalize_1st_derivative = normalize_1st_derivative
        self.keep_2nd_derivative = keep_2nd_derivative
        self.normalize_2nd_derivative = normalize_2nd_derivative
        if self.keep_2nd_derivative:
            self.keep_1st_derivative=self.keep_2nd_derivative
            self.normalize_1st_derivative=self.normalize_2nd_derivative
        self.scaler = None
        # Select appropriate scalers
        if scaler_type == 'robust':
            scalers = [
            serialize(preprocessing.RobustScaler),
        ]
            self.scaler = TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in scalers])
        elif scaler_type == 'linear':
            scalers = [
                serialize(preprocessing.MinMaxScaler, [(-1, 1)]),
            ]
            self.scaler = TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in scalers])
        elif scaler_type == 'log':
            scalers = [
                serialize(LogTransformer),
                serialize(preprocessing.MinMaxScaler, [(-1, 1)]),
            ]
            self.scaler = TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in scalers])
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.data_len = 0
        self.data_channel = 1

    def fit(self, X):
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array with shape (num_samples, len_data)")

        self.data_len = X.shape[1]

        if self.scaler_scope == 'wholeband':
            flattened_data = X.reshape(-1, 1)
            self.scaler.fit(flattened_data)
        elif self.scaler_scope == 'wavelength':
            self.scaler.fit(X)
        else:
            raise ValueError(f"Unknown scaler scope: {self.scaler_scope}")


    def transform(self, X):
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array with shape (num_samples, len_data)")

        if self.scaler_scope == 'wholeband':
            flattened_data = X.reshape(-1, 1)
            scaled_flattened_data = self.scaler.transform(flattened_data)
            # it used to be scaled_flattened_data = self.scaler.fit_transform(flattened_data)
            # but this caused severe mistakes.
            X_transformed=scaled_flattened_data.reshape(X.shape)
        elif self.scaler_scope == 'wavelength':
            X_transformed = self.scaler.transform(X)
        else:
            raise ValueError(f"Unknown scaler scope: {self.scaler_scope}")

        channels = [X_transformed]
        if self.keep_shape:
            scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(X.shape[0])]
            X_curveShape = np.array(
                [scalers[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in
                 enumerate(X)])
            channels.append(X_curveShape)

        if self.keep_1st_derivative:
            X_derivative_1st = np.diff(X, axis=1, prepend=X[:, [0]])
            if self.normalize_1st_derivative:
                scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(X_derivative_1st.shape[0])]
                X_derivative_1st = np.array(
                    [scalers[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in
                     enumerate(X_derivative_1st)])
            channels.append(X_derivative_1st)

        if self.keep_2nd_derivative:
            X_derivative_2nd = np.diff(X_derivative_1st, axis=1, prepend=X_derivative_1st[:, [0]])
            if self.normalize_2nd_derivative:
                scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(X_derivative_2nd.shape[0])]
                X_derivative_2nd = np.array(
                    [scalers[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in
                     enumerate(X_derivative_2nd)])
            channels.append(X_derivative_2nd)

        X_combined = np.stack(channels, axis=1)
        self.data_channel = len(channels)
        return X_combined

    def inverse_transform(self, X):
        if self.keep_1st_derivative or self.keep_2nd_derivative or self.keep_shape:
            X_original = X[:, 0, :]
        else:
            X_original = X
        X_original = X_original.squeeze()
        if self.scaler_scope == 'wholeband':
            flattened_data = X_original.reshape(-1, 1)
            inversed_flattened_data = self.scaler.inverse_transform(flattened_data)
            X_inversed = inversed_flattened_data.reshape(X_original.shape)
        elif self.scaler_scope == 'wavelength':
            X_inversed = self.scaler.inverse_transform(X_original)
        else:
            raise ValueError(f"Unknown scaler scope: {self.scaler_scope}")

        return X_inversed