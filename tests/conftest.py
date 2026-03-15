import pytest
import numpy as np

@pytest.fixture
def signal_sana():
    """Seno puro 50Hz — máquina en buen estado"""
    t = np.linspace(0, 1, 1000)
    return np.sin(2 * np.pi * 50 * t)

@pytest.fixture
def signal_con_falla():
    """Seno con impacto — simula falla de rodamiento"""
    t      = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 50 * t).copy()
    signal[500] = 10.0
    return signal

@pytest.fixture
def signal_constante():
    """Señal plana — para casos límite"""
    return np.full(100, 2.0)

@pytest.fixture
def signal_corta():
    """Señal mínima válida"""
    return np.array([1.0, -1.0, 0.5, -0.5])
