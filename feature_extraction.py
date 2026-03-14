import numpy as np

def compute_rms(signal):
    """
    Calcula la raíz cuadrada de la media de los cuadrados.

    >>> compute_rms([1.0, -1.0, 1.0, -1.0])
    1.0
    >>> compute_rms([])
    Traceback (most recent call last):
        ...
    ValueError: La señal no puede estar vacía
    """
    if signal is None:
        raise TypeError("signal no puede ser None")
    if len(signal) == 0:
        raise ValueError("La señal no puede estar vacía")
    arr = np.array(signal, dtype=float)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("La señal contiene NaN o infinitos")
    return float(np.sqrt(np.mean(arr ** 2)))


def compute_peak(signal):
    """
    Retorna el valor absoluto máximo de la señal.

    >>> compute_peak([0.5, -1.8, 0.9])
    1.8
    >>> compute_peak([])
    Traceback (most recent call last):
        ...
    ValueError: La señal no puede estar vacía
    """
    if signal is None:
        raise TypeError("signal no puede ser None")
    if len(signal) == 0:
        raise ValueError("La señal no puede estar vacía")
    arr = np.array(signal, dtype=float)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("La señal contiene NaN o infinitos")
    return float(np.max(np.abs(arr)))


def compute_kurtosis(signal):
    """
    Calcula la kurtosis de Pearson.
    Referencia: señal gaussiana = 3.0

    >>> compute_kurtosis([2.0, 2.0, 2.0])
    Traceback (most recent call last):
        ...
    ValueError: La señal es constante, std=0
    """
    if signal is None:
        raise TypeError("signal no puede ser None")
    if len(signal) == 0:
        raise ValueError("La señal no puede estar vacía")
    arr = np.array(signal, dtype=float)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("La señal contiene NaN o infinitos")
    std = np.std(arr)
    if std == 0:
        raise ValueError("La señal es constante, std=0")
    media = np.mean(arr)
    return float(np.mean(((arr - media) / std) ** 4))


def compute_variance(signal):
    """
    Calcula la varianza de la señal.

    >>> compute_variance([2.0, 2.0, 2.0])
    0.0
    >>> compute_variance([])
    Traceback (most recent call last):
        ...
    ValueError: La señal no puede estar vacía
    """
    if signal is None:
        raise TypeError("signal no puede ser None")
    if len(signal) == 0:
        raise ValueError("La señal no puede estar vacía")
    arr = np.array(signal, dtype=float)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("La señal contiene NaN o infinitos")
    return float(np.var(arr))
