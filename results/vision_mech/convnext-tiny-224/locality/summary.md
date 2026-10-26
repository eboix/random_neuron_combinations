# Vision Experiment B — facebook/convnext-tiny-224

Reproduction: our CLIP top-16 coherence vs published, 128 units: Pearson r = 0.985, mean abs diff 0.010

## Median per condition

```
                 clip_coh                  coh_excess                          pr                      purity              
layer_name  stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
cond                                                                                                                       
neuron              0.675         0.689         0.269         0.223        10.523        13.072         0.125         0.250
related_5           0.650         0.670         0.318         0.230         8.183        12.694         0.125         0.250
related_20          0.658         0.641         0.325         0.206         8.068        14.969         0.125         0.125
ksparse_2           0.651         0.685         0.272         0.198        10.704        15.382         0.125         0.188
ksparse_5           0.667         0.704         0.248         0.190        11.315        15.182         0.125         0.312
ksparse_20          0.657         0.681         0.248         0.155        11.318        17.265         0.125         0.375
ksparse_100         0.664         0.633         0.247         0.097        10.565        24.586         0.125         0.188
dense_raw           0.660         0.610         0.151         0.091        18.145        30.917         0.062         0.062
dense_std           0.655         0.613         0.164         0.090        14.807        31.676         0.125         0.062
xdir                0.648         0.604         0.141         0.096        22.606        36.961         0.062         0.062
```

```
                  cos_win                     dom_eff                   dom_share              
layer_name  stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
cond                                                                                           
neuron              13.26         12.43           NaN           NaN           NaN           NaN
related_5             NaN           NaN          1.61          1.64          0.82          0.84
related_20            NaN           NaN          3.16          4.53          0.56          0.43
ksparse_2             NaN           NaN          1.00          1.00          1.00          1.00
ksparse_5             NaN           NaN          1.00          1.24          1.00          0.94
ksparse_20            NaN           NaN          1.41          2.11          0.90          0.67
ksparse_100           NaN           NaN          2.21          6.63          0.76          0.34
dense_raw             NaN           NaN         18.58         37.82          0.16          0.10
dense_std             NaN           NaN         13.23         41.86          0.27          0.10
```

## Mediation by coh_excess: clip_coh, effects vs neuron (90% bootstrap CI)

```
             effect_total  total_lo  total_hi  effect_direct  direct_lo  direct_hi  attenuation
cond                                                                                           
related_5          -0.020    -0.035    -0.005         -0.029     -0.043     -0.014       -0.409
related_20         -0.028    -0.042    -0.012         -0.035     -0.048     -0.020       -0.242
ksparse_2          -0.009    -0.025     0.007          0.007     -0.009      0.021        1.720
ksparse_5          -0.002    -0.018     0.014          0.009     -0.005      0.024        5.179
ksparse_20         -0.010    -0.026     0.006          0.005     -0.009      0.019        1.463
ksparse_100        -0.032    -0.048    -0.018         -0.006     -0.020      0.009        0.821
dense_raw          -0.051    -0.064    -0.037         -0.008     -0.021      0.006        0.842
dense_std          -0.048    -0.063    -0.034         -0.011     -0.024      0.002        0.771
```

## Mediation by coh_excess: purity, effects vs neuron (90% bootstrap CI)

```
             effect_total  total_lo  total_hi  effect_direct  direct_lo  direct_hi  attenuation
cond                                                                                           
related_5          -0.043    -0.091     0.005         -0.060     -0.105     -0.015       -0.404
related_20         -0.105    -0.150    -0.060         -0.119     -0.163     -0.077       -0.133
ksparse_2          -0.056    -0.107    -0.006         -0.023     -0.073      0.024        0.579
ksparse_5           0.016    -0.034     0.068          0.040     -0.008      0.089       -1.558
ksparse_20          0.010    -0.043     0.062          0.041     -0.010      0.089       -3.056
ksparse_100        -0.043    -0.090     0.003          0.012     -0.034      0.057        1.284
dense_raw          -0.170    -0.212    -0.129         -0.082     -0.125     -0.040        0.520
dense_std          -0.167    -0.210    -0.126         -0.089     -0.133     -0.048        0.465
```

## Within-condition Spearman

```
         x        y   n    rho     lo     hi
coh_excess clip_coh 900  0.365  0.313  0.416
coh_excess   purity 900  0.139  0.084  0.193
        pr clip_coh 900 -0.396 -0.447 -0.346
        pr   purity 900 -0.213 -0.264 -0.160
```

## Robustness: clip_coh ~ cond + layer + covariates

```
spec                              +coh               +purity           +purity+coh                  cond
term                                                                                                    
coh_excess        +0.323 [+0.28,+0.37]                   NaN  +0.215 [+0.17,+0.26]                   NaN
cond=dense_raw    -0.008 [-0.02,+0.01]  -0.017 [-0.03,-0.01]  +0.005 [-0.01,+0.02]  -0.051 [-0.06,-0.04]
cond=dense_std    -0.011 [-0.03,+0.00]  -0.016 [-0.03,-0.00]  +0.003 [-0.01,+0.01]  -0.048 [-0.06,-0.03]
cond=ksparse_100  -0.006 [-0.02,+0.01]  -0.024 [-0.04,-0.01]  -0.008 [-0.02,+0.01]  -0.032 [-0.05,-0.02]
cond=ksparse_2    +0.007 [-0.01,+0.02]  +0.002 [-0.01,+0.01]  +0.010 [-0.00,+0.02]  -0.009 [-0.02,+0.01]
cond=ksparse_20   +0.005 [-0.01,+0.02]  -0.012 [-0.02,+0.00]  -0.002 [-0.01,+0.01]  -0.010 [-0.03,+0.01]
cond=ksparse_5    +0.009 [-0.01,+0.02]  -0.005 [-0.02,+0.01]  +0.003 [-0.01,+0.02]  -0.002 [-0.02,+0.01]
cond=related_20   -0.035 [-0.05,-0.02]  -0.007 [-0.02,+0.00]  -0.015 [-0.03,-0.00]  -0.028 [-0.04,-0.01]
cond=related_5    -0.029 [-0.04,-0.01]  -0.012 [-0.02,+0.00]  -0.019 [-0.03,-0.01]  -0.020 [-0.04,-0.00]
purity                             NaN  +0.195 [+0.18,+0.21]  +0.160 [+0.14,+0.18]                   NaN
```