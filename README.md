# Latent-ODE-with-various-encoder
A time series forcasting model based on Latent-ODE.

* Physionet (discretization by 3.6 min, attn encoder)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder attn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.06 --classif
```

* Physionet (discretization by 1 min, mamba encoder)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder mamba --gen-layers 3 --units 50 --quantization 0.016 --classif
```