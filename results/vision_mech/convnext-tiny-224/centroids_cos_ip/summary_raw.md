# Centroid test (raw MLP inputs)

Assignment `cos` = argmax cosine with unit centroids; `euclid` = nearest raw-scale centroid in raw (centered) x; `ip` = largest inner product with rows at native norm (largest pre-activation). `obj` = mean max-cosine of held-out positions to the set (cos only); `obj_norm` = 0 at the covariance-matched random null, 1 at spherical k-means; `dist` = mean squared distance to the nearest centroid / total variance (euclid only); `ve` / `ve_raw` = variance of the unit-normalized / raw x explained by the partition (1 - within-cluster SS / total SS); `nmi_*` = NMI of the partition with ImageNet class / spatial position, minus the NMI of a size-matched random partition.

## cos, trained, M = 32

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -0.327        -0.561         0.050         0.038         0.051         0.038         0.017         0.060         0.021         0.030
rand_cov          0.000         0.000         0.056         0.084         0.058         0.083         0.020         0.043         0.034         0.132
rand_rot         -0.289        -0.359         0.032         0.040         0.034         0.041         0.014         0.047         0.019         0.043
combo_5          -0.132        -0.088         0.037         0.034         0.037         0.036         0.012         0.023         0.014         0.042
data_pts          0.404         0.614         0.073         0.099         0.075         0.100         0.032         0.071         0.049         0.154
kmeans            1.000         1.000         0.114         0.140         0.119         0.141         0.060         0.170         0.064         0.169
rand_iso         -0.290        -0.355         0.031         0.039         0.032         0.040         0.013         0.051         0.019         0.041
rand_gcov        -0.352        -0.545         0.039         0.050         0.040         0.052         0.015         0.043         0.019         0.051
combo_2          -0.143        -0.099         0.034         0.062         0.034         0.063         0.012         0.030         0.013         0.080
combo_20         -0.140        -0.044         0.034         0.034         0.035         0.035         0.013         0.017         0.016         0.046
```

## cos, trained, M = 128

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -0.332        -0.580         0.088         0.071         0.090         0.071         0.015         0.070         0.022         0.038
rand_cov          0.000         0.000         0.083         0.103         0.085         0.103         0.017         0.045         0.035         0.135
rand_rot         -0.356        -0.415         0.050         0.060         0.052         0.063         0.009         0.044         0.021         0.054
combo_5          -0.179        -0.080         0.057         0.070         0.058         0.071         0.012         0.031         0.016         0.078
data_pts          0.457         0.620         0.116         0.134         0.120         0.135         0.027         0.096         0.048         0.141
kmeans            1.000         1.000         0.163         0.181         0.169         0.180         0.040         0.175         0.058         0.152
rand_iso         -0.353        -0.408         0.050         0.058         0.052         0.059         0.009         0.046         0.018         0.051
rand_gcov        -0.416        -0.598         0.061         0.073         0.062         0.075         0.011         0.040         0.019         0.063
combo_2          -0.194        -0.107         0.059         0.079         0.060         0.080         0.011         0.033         0.015         0.091
combo_20         -0.163        -0.076         0.055         0.056         0.056         0.056         0.012         0.030         0.016         0.064
```

## cos, trained, M = 512

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -0.344        -0.560         0.137         0.134         0.140         0.132         0.009         0.088         0.022         0.037
rand_cov          0.000         0.000         0.119         0.130         0.121         0.132         0.010         0.039         0.029         0.123
rand_rot         -0.427        -0.441         0.082         0.090         0.084         0.093         0.006         0.036         0.015         0.054
combo_5          -0.223        -0.075         0.093         0.089         0.093         0.093         0.009         0.036         0.016         0.085
data_pts          0.531         0.668         0.175         0.194         0.179         0.193         0.015         0.104         0.039         0.110
kmeans            1.000         1.000         0.213         0.241         0.220         0.238         0.021         0.162         0.039         0.110
rand_iso         -0.416        -0.427         0.083         0.087         0.084         0.089         0.005         0.035         0.015         0.052
rand_gcov        -0.474        -0.597         0.095         0.099         0.096         0.103         0.006         0.032         0.014         0.060
combo_2          -0.234        -0.103         0.096         0.094         0.097         0.098         0.007         0.037         0.017         0.086
combo_20         -0.215        -0.086         0.086         0.077         0.086         0.078         0.009         0.037         0.017         0.074
```

## cos, trained, M = 1536

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate             -0.336         0.201         0.204         0.007         0.020
rand_cov          0.000         0.169         0.169         0.008         0.022
rand_rot         -0.458         0.132         0.134         0.003         0.011
combo_5          -0.249         0.138         0.138         0.005         0.013
data_pts          0.617         0.244         0.248         0.010         0.024
kmeans            1.000         0.267         0.272         0.013         0.025
rand_iso         -0.459         0.132         0.134         0.003         0.010
rand_gcov        -0.510         0.145         0.145         0.004         0.012
combo_2          -0.265         0.150         0.151         0.005         0.015
combo_20         -0.234         0.127         0.127         0.006         0.013
```

## cos, trained, M = 3072

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate             -0.538         0.319         0.315         0.106         0.028
rand_cov          0.000         0.190         0.192         0.032         0.103
rand_rot         -0.500         0.167         0.174         0.024         0.043
combo_5          -0.105         0.124         0.131         0.037         0.080
data_pts          0.766         0.355         0.350         0.093         0.055
kmeans            1.000         0.366         0.359         0.125         0.065
rand_iso         -0.485         0.171         0.178         0.023         0.045
rand_gcov        -0.649         0.161         0.171         0.023         0.056
combo_2          -0.136         0.139         0.147         0.032         0.076
combo_20         -0.109         0.097         0.102         0.040         0.066
```

## cos, random, M = 32

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -1.798        -1.310         0.200         0.113         0.200         0.113         0.010         0.008         0.026         0.026
rand_cov          0.000         0.000         0.293         0.195         0.293         0.195         0.016         0.018         0.046         0.063
rand_rot         -1.812        -1.304         0.197         0.114         0.197         0.114         0.009         0.010         0.025         0.026
combo_5          -1.809        -1.311         0.201         0.112         0.201         0.112         0.010         0.008         0.025         0.025
data_pts          0.302         0.251         0.294         0.205         0.294         0.205         0.015         0.018         0.049         0.104
kmeans            1.000         1.000         0.354         0.253         0.354         0.253         0.020         0.022         0.100         0.169
rand_iso         -1.810        -1.313         0.195         0.112         0.195         0.112         0.010         0.009         0.024         0.026
rand_gcov        -1.801        -1.303         0.202         0.117         0.202         0.117         0.010         0.009         0.024         0.024
combo_2          -1.814        -1.311         0.197         0.115         0.197         0.115         0.009         0.009         0.025         0.024
combo_20         -1.801        -1.306         0.204         0.115         0.204         0.115         0.010         0.009         0.025         0.024
```

## cos, random, M = 128

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -2.548        -1.643         0.255         0.154         0.255         0.154         0.007         0.006         0.033         0.043
rand_cov          0.000         0.000         0.348         0.243         0.348         0.243         0.011         0.013         0.061         0.100
rand_rot         -2.551        -1.637         0.251         0.158         0.251         0.158         0.007         0.006         0.035         0.046
combo_5          -2.550        -1.640         0.255         0.155         0.255         0.155         0.007         0.006         0.033         0.044
data_pts          0.322         0.305         0.361         0.257         0.361         0.257         0.012         0.014         0.090         0.166
kmeans            1.000         1.000         0.410         0.307         0.410         0.307         0.011         0.012         0.125         0.228
rand_iso         -2.541        -1.642         0.257         0.154         0.257         0.154         0.008         0.006         0.036         0.043
rand_gcov        -2.542        -1.638         0.255         0.157         0.255         0.157         0.008         0.006         0.033         0.043
combo_2          -2.556        -1.641         0.253         0.156         0.253         0.156         0.007         0.006         0.034         0.044
combo_20         -2.548        -1.639         0.253         0.157         0.253         0.157         0.007         0.006         0.036         0.044
```

## cos, random, M = 512

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -3.109        -2.039         0.307         0.203         0.307         0.203         0.004         0.004         0.035         0.050
rand_cov          0.000         0.000         0.400         0.291         0.400         0.291         0.007         0.007         0.061         0.112
rand_rot         -3.103        -2.039         0.306         0.204         0.306         0.204         0.004         0.003         0.034         0.053
combo_5          -3.110        -2.043         0.308         0.201         0.308         0.202         0.004         0.004         0.035         0.052
data_pts          0.476         0.420         0.421         0.313         0.421         0.313         0.007         0.008         0.108         0.200
kmeans            1.000         1.000         0.454         0.351         0.454         0.351         0.005         0.006         0.119         0.234
rand_iso         -3.095        -2.038         0.306         0.205         0.306         0.205         0.004         0.004         0.035         0.053
rand_gcov        -3.105        -2.036         0.307         0.204         0.307         0.204         0.004         0.003         0.035         0.051
combo_2          -3.109        -2.040         0.305         0.204         0.305         0.204         0.005         0.003         0.031         0.052
combo_20         -3.097        -2.042         0.307         0.204         0.307         0.204         0.005         0.003         0.034         0.053
```

## cos, random, M = 1536

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate             -3.536         0.366         0.366         0.002         0.031
rand_cov          0.000         0.454         0.454         0.004         0.056
rand_rot         -3.520         0.365         0.365         0.003         0.030
combo_5          -3.538         0.365         0.365         0.003         0.032
data_pts          0.616         0.477         0.477         0.004         0.096
kmeans            1.000         0.493         0.493         0.003         0.102
rand_iso         -3.534         0.365         0.365         0.003         0.031
rand_gcov        -3.534         0.366         0.366         0.003         0.032
combo_2          -3.545         0.364         0.364         0.003         0.028
combo_20         -3.531         0.365         0.365         0.003         0.029
```

## cos, random, M = 3072

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate             -2.602         0.320         0.320         0.002         0.043
rand_cov          0.000         0.388         0.388         0.004         0.101
rand_rot         -2.604         0.320         0.320         0.001         0.044
combo_5          -2.604         0.319         0.319         0.002         0.043
data_pts          0.647         0.413         0.413         0.003         0.193
kmeans            1.000         0.417         0.417         0.003         0.209
rand_iso         -2.611         0.318         0.318         0.001         0.042
rand_gcov        -2.610         0.320         0.320         0.001         0.045
combo_2          -2.609         0.319         0.319         0.002         0.044
combo_20         -2.612         0.317         0.317         0.002         0.043
```

## ip, trained, M = 32

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.050         0.037         0.051         0.037        27.985        22.023         0.017         0.059         0.022         0.028
rand_rot          0.032         0.040         0.033         0.041        23.805        15.220         0.014         0.047         0.018         0.043
combo_5           0.036         0.032         0.036         0.033        17.494         4.350         0.012         0.024         0.014         0.041
rand_iso          0.031         0.039         0.031         0.040        24.049        18.100         0.013         0.050         0.018         0.042
rand_gcov         0.039         0.050         0.040         0.052        23.614        15.917         0.015         0.043         0.019         0.052
combo_2           0.034         0.062         0.034         0.062        16.857         6.688         0.011         0.031         0.012         0.081
combo_20          0.033         0.031         0.034         0.033        14.307         3.555         0.013         0.015         0.016         0.038
```

## ip, trained, M = 128

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.088         0.071         0.091         0.070        98.172        58.415         0.016         0.070         0.024         0.037
rand_rot          0.049         0.058         0.050         0.060        75.578        47.146         0.010         0.045         0.019         0.052
combo_5           0.055         0.069         0.056         0.070        48.193         8.790         0.012         0.031         0.015         0.080
rand_iso          0.049         0.057         0.051         0.059        77.475        45.450         0.010         0.047         0.019         0.050
rand_gcov         0.061         0.074         0.062         0.076        78.117        39.848         0.012         0.039         0.020         0.063
combo_2           0.057         0.076         0.058         0.078        52.111        11.598         0.011         0.034         0.014         0.087
combo_20          0.052         0.052         0.053         0.052        40.126         6.165         0.012         0.026         0.014         0.054
```

## ip, trained, M = 512

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.139         0.137         0.142         0.135       338.248       275.134         0.010         0.088         0.025         0.034
rand_rot          0.079         0.090         0.081         0.093       221.181       143.347         0.007         0.036         0.015         0.053
combo_5           0.084         0.084         0.086         0.086       112.678        18.036         0.010         0.034         0.019         0.084
rand_iso          0.079         0.085         0.080         0.088       237.883       142.296         0.006         0.035         0.016         0.052
rand_gcov         0.094         0.100         0.095         0.104       258.883        89.647         0.007         0.033         0.013         0.060
combo_2           0.089         0.091         0.090         0.095       154.155        21.495         0.009         0.034         0.020         0.088
combo_20          0.079         0.076         0.079         0.078       111.441        16.212         0.010         0.037         0.019         0.073
```

## ip, trained, M = 1536

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate              0.204         0.207       865.883         0.007         0.022
rand_rot          0.122         0.123       525.509         0.004         0.013
combo_5           0.122         0.122       293.087         0.006         0.012
rand_iso          0.120         0.121       458.465         0.005         0.013
rand_gcov         0.141         0.141       568.129         0.004         0.011
combo_2           0.133         0.135       368.407         0.007         0.014
combo_20          0.116         0.115       248.323         0.006         0.012
```

## ip, trained, M = 3072

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate              0.324         0.318      1765.496         0.106         0.023
rand_rot          0.157         0.164       469.829         0.024         0.045
combo_5           0.112         0.118        44.315         0.036         0.081
rand_iso          0.159         0.165       462.939         0.026         0.046
rand_gcov         0.161         0.171       207.533         0.023         0.056
combo_2           0.124         0.131        56.588         0.032         0.080
combo_20          0.095         0.099        44.044         0.040         0.068
```

## ip, random, M = 32

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.199         0.113         0.199         0.113        26.847        29.057         0.009         0.008         0.026         0.025
rand_rot          0.197         0.114         0.197         0.114        26.594        29.158         0.010         0.009         0.025         0.026
combo_5           0.201         0.112         0.201         0.112        26.313        28.588         0.010         0.009         0.026         0.025
rand_iso          0.194         0.112         0.194         0.112        26.832        28.906         0.010         0.008         0.025         0.026
rand_gcov         0.202         0.117         0.202         0.117        26.435        28.183         0.010         0.008         0.024         0.024
combo_2           0.197         0.115         0.197         0.115        27.169        29.176         0.009         0.010         0.024         0.025
combo_20          0.203         0.115         0.203         0.115        25.502        28.255         0.010         0.009         0.024         0.024
```

## ip, random, M = 128

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.254         0.153         0.254         0.153        84.655       102.935         0.007         0.006         0.033         0.042
rand_rot          0.250         0.158         0.250         0.158        82.588       100.111         0.007         0.006         0.035         0.046
combo_5           0.254         0.155         0.254         0.155        81.162        99.663         0.007         0.006         0.032         0.044
rand_iso          0.256         0.154         0.256         0.154        82.137       101.413         0.008         0.006         0.036         0.042
rand_gcov         0.254         0.157         0.254         0.157        80.427        98.708         0.008         0.006         0.033         0.043
combo_2           0.251         0.156         0.251         0.156        84.665       100.840         0.007         0.006         0.035         0.044
combo_20          0.252         0.157         0.252         0.157        85.229       100.910         0.007         0.006         0.036         0.044
```

## ip, random, M = 512

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.305         0.202         0.305         0.202       238.481       311.296         0.005         0.004         0.035         0.049
rand_rot          0.305         0.204         0.305         0.204       238.832       327.216         0.004         0.004         0.034         0.053
combo_5           0.306         0.201         0.306         0.201       248.645       337.705         0.004         0.004         0.034         0.052
rand_iso          0.305         0.204         0.305         0.204       246.231       311.507         0.004         0.004         0.032         0.054
rand_gcov         0.306         0.203         0.306         0.203       249.446       314.293         0.004         0.004         0.033         0.051
combo_2           0.303         0.203         0.303         0.203       234.867       316.359         0.004         0.003         0.033         0.051
combo_20          0.306         0.204         0.306         0.204       244.544       329.797         0.005         0.004         0.033         0.052
```

## ip, random, M = 1536

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate              0.362         0.362       567.010         0.003         0.031
rand_rot          0.362         0.362       528.632         0.003         0.030
combo_5           0.362         0.362       557.953         0.003         0.031
rand_iso          0.361         0.361       504.490         0.004         0.031
rand_gcov         0.362         0.362       543.622         0.003         0.032
combo_2           0.360         0.360       554.051         0.003         0.031
combo_20          0.361         0.361       538.448         0.003         0.030
```

## ip, random, M = 3072

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate              0.317         0.317      1192.366         0.002         0.044
rand_rot          0.317         0.317      1364.966         0.001         0.044
combo_5           0.316         0.316      1311.068         0.002         0.043
rand_iso          0.316         0.316      1425.182         0.001         0.043
rand_gcov         0.316         0.316      1336.491         0.002         0.046
combo_2           0.315         0.315      1280.405         0.002         0.043
combo_20          0.315         0.315      1382.313         0.001         0.044
```
