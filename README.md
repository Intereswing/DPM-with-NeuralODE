# Latent-ODE-with-various-encoder
A time series forcasting model based on Latent-ODE.

* Physionet (discretization by 3.6 min, attn encoder)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder attn --gen-layers 3 --units 50 --quantization 0.06 --classif
```

* Physionet (discretization by 3.6 min, ode-mamba encoder)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder mamba --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.06 --classif
```

* Physionet (discretization by 3.6 min, ode-lstm encoder)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder odernn_lstm --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.06 --classif
```

* Physionet (discretization by 3.6 min, ode-gru encoder)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder odernn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.06 --classif
```

* Physionet (discretization by 3.6 min, rnn encoder)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder rnn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.06 --classif
```

* Physionet (discretization by 3.6 min, rnn vae)
```
python run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --encoder rnn --decoder rnn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.06 --classif
```




* Human Activity (attn encoder)
```
python run_models.py --niters 200 -n 10000 -l 15 --dataset activity --latent-ode --encoder attn --gen-layers 2 --units 500 --classif  --linear-classif

```

* Human Activity (ode-mamba encoder)
```
python run_models.py --niters 200 -n 10000 -l 15 --dataset activity --latent-ode --encoder mamba --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 500 --gru-units 50 --classif  --linear-classif
```

* Human Activity (ode-lstm encoder)
```
python run_models.py --niters 200 -n 10000 -l 15 --dataset activity --latent-ode --encoder odernn_lstm --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 500 --gru-units 50 --classif  --linear-classif
```

* Human Activity (ode-rnn encoder)
```
python run_models.py --niters 200 -n 10000 -l 15 --dataset activity --latent-ode --encoder odernn --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 500 --gru-units 50 --classif  --linear-classif
```

* Human Activity (rnn encoder)
```
python run_models.py --niters 200 -n 10000 -l 15 --dataset activity --latent-ode --encoder rnn --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 500 --gru-units 50 --classif  --linear-classif
```

* Human Activity (rnn vae)
```
python run_models.py --niters 200 -n 10000 -l 15 --dataset activity --latent-ode --encoder rnn --decoder rnn --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 500 --gru-units 50 --classif  --linear-classif
```
