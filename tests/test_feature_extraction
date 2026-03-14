import pytest
import numpy as np
from feature_extraction import (
    compute_rms,
    compute_peak,
    compute_kurtosis,
    compute_variance
)

# ══════════════════════════════════════════
# 1. PRUEBAS BÁSICAS
# ══════════════════════════════════════════

class TestComputeRMS:

    def test_señal_simetrica(self):
        """Oráculo 1: calculado a mano"""
        assert compute_rms([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0, abs=1e-6)

    def test_seno_puro(self, signal_sana):
        """Oráculo 3: RMS teórico seno = A/√2 = 0.7071"""
        assert compute_rms(signal_sana) == pytest.approx(0.7071, abs=1e-3)

    def test_vs_numpy(self, signal_sana):
        """Oráculo 4: comparar contra numpy"""
        esperado = np.sqrt(np.mean(signal_sana ** 2))
        assert compute_rms(signal_sana) == pytest.approx(esperado, rel=1e-6)

    def test_señal_ceros(self):
        """Caso límite: señal de ceros"""
        assert compute_rms([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_señal_vacia(self):
        """Entrada inválida: señal vacía"""
        with pytest.raises(ValueError):
            compute_rms([])

    def test_contiene_nan(self):
        """Entrada inválida: contiene NaN"""
        with pytest.raises(ValueError):
            compute_rms([1.0, float('nan'), 2.0])

    def test_tipo_none(self):
        """Entrada inválida: None"""
        with pytest.raises(TypeError):
            compute_rms(None)


class TestComputePeak:

    def test_valores_positivos(self):
        assert compute_peak([0.5, 1.8, 0.9]) == pytest.approx(1.8)

    def test_valores_negativos(self):
        """Peak toma valor absoluto"""
        assert compute_peak([-0.5, -1.8, -0.9]) == pytest.approx(1.8)

    def test_valores_mixtos(self):
        assert compute_peak([1.0, -3.0, 2.0]) == pytest.approx(3.0)

    def test_un_solo_valor(self):
        """Caso límite: un elemento"""
        assert compute_peak([-4.5]) == pytest.approx(4.5)

    def test_señal_vacia(self):
        with pytest.raises(ValueError):
            compute_peak([])


class TestComputeKurtosis:

    def test_señal_gaussiana(self):
        """Oráculo 3: kurtosis gaussiana teórica = 3.0"""
        np.random.seed(42)
        signal = np.random.normal(0, 1, 10000)
        assert compute_kurtosis(signal) == pytest.approx(3.0, abs=0.1)

    def test_señal_falla_mayor_que_sana(self, signal_sana, signal_con_falla):
        """Señal con impacto debe tener kurtosis mayor"""
        assert compute_kurtosis(signal_con_falla) > compute_kurtosis(signal_sana)

    def test_señal_constante(self, signal_constante):
        """Caso límite + inválida: std=0"""
        with pytest.raises(ValueError):
            compute_kurtosis(signal_constante)

    def test_señal_vacia(self):
        with pytest.raises(ValueError):
            compute_kurtosis([])


class TestComputeVariance:

    def test_señal_constante_varianza_cero(self, signal_constante):
        """Señal constante no tiene dispersión"""
        assert compute_variance(signal_constante) == pytest.approx(0.0)

    def test_señal_simetrica(self):
        """Oráculo 1: calculado a mano"""
        assert compute_variance([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0, abs=1e-6)

    def test_señal_vacia(self):
        with pytest.raises(ValueError):
            compute_variance([])
