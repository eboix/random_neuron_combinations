# Centroid test (raw MLP inputs)

Assignment `cos` = argmax cosine with unit centroids; `euclid` = nearest raw-scale centroid in raw (centered) x. `obj` = mean max-cosine of held-out positions to the set (cos only); `obj_norm` = 0 at the covariance-matched random null, 1 at spherical k-means; `dist` = mean squared distance to the nearest centroid / total variance (euclid only); `ve` / `ve_raw` = variance of the unit-normalized / raw x explained by the partition (1 - within-cluster SS / total SS); `nmi_*` = NMI of the partition with ImageNet class / spatial position, minus the NMI of a size-matched random partition.

## cos, trained, M = 32

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -0.327        -0.561         0.050         0.038         0.051         0.038         0.017         0.060         0.021         0.030
rand_cov         0.000         0.000         0.056         0.084         0.058         0.083         0.020         0.043         0.034         0.132
rand_rot        -0.289        -0.359         0.032         0.040         0.034         0.041         0.014         0.047         0.019         0.043
combo_5         -0.132        -0.088         0.037         0.034         0.037         0.036         0.012         0.023         0.014         0.042
data_pts         0.404         0.614         0.073         0.099         0.075         0.100         0.032         0.071         0.049         0.154
kmeans           1.000         1.000         0.114         0.140         0.119         0.141         0.060         0.170         0.064         0.169
rand_iso        -0.290        -0.355         0.031         0.039         0.032         0.040         0.013         0.051         0.019         0.041
combo_2         -0.143        -0.099         0.034         0.062         0.034         0.063         0.012         0.030         0.013         0.080
combo_20        -0.140        -0.044         0.034         0.034         0.035         0.035         0.013         0.017         0.016         0.046
```

## cos, trained, M = 128

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -0.332        -0.580         0.088         0.071         0.090         0.071         0.015         0.070         0.022         0.038
rand_cov         0.000         0.000         0.083         0.103         0.085         0.103         0.017         0.045         0.035         0.135
rand_rot        -0.356        -0.415         0.050         0.060         0.052         0.063         0.009         0.044         0.021         0.054
combo_5         -0.179        -0.080         0.057         0.070         0.058         0.071         0.012         0.031         0.016         0.078
data_pts         0.457         0.620         0.116         0.134         0.120         0.135         0.027         0.096         0.048         0.141
kmeans           1.000         1.000         0.163         0.181         0.169         0.180         0.040         0.175         0.058         0.152
rand_iso        -0.353        -0.408         0.050         0.058         0.052         0.059         0.009         0.046         0.018         0.051
combo_2         -0.194        -0.107         0.059         0.079         0.060         0.080         0.011         0.033         0.015         0.091
combo_20        -0.163        -0.076         0.055         0.056         0.056         0.056         0.012         0.030         0.016         0.064
```

## cos, trained, M = 512

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -0.344        -0.560         0.137         0.134         0.140         0.132         0.009         0.088         0.022         0.037
rand_cov         0.000         0.000         0.119         0.130         0.121         0.132         0.010         0.039         0.029         0.123
rand_rot        -0.427        -0.441         0.082         0.090         0.084         0.093         0.006         0.036         0.015         0.054
combo_5         -0.223        -0.075         0.093         0.089         0.093         0.093         0.009         0.036         0.016         0.085
data_pts         0.531         0.668         0.175         0.194         0.179         0.193         0.015         0.104         0.039         0.110
kmeans           1.000         1.000         0.213         0.241         0.220         0.238         0.021         0.162         0.039         0.110
rand_iso        -0.416        -0.427         0.083         0.087         0.084         0.089         0.005         0.035         0.015         0.052
combo_2         -0.234        -0.103         0.096         0.094         0.097         0.098         0.007         0.037         0.017         0.086
combo_20        -0.215        -0.086         0.086         0.077         0.086         0.078         0.009         0.037         0.017         0.074
```

## cos, trained, M = 1536

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate            -0.336         0.201         0.204         0.007         0.020
rand_cov         0.000         0.169         0.169         0.008         0.022
rand_rot        -0.458         0.132         0.134         0.003         0.011
combo_5         -0.249         0.138         0.138         0.005         0.013
data_pts         0.617         0.244         0.248         0.010         0.024
kmeans           1.000         0.267         0.272         0.013         0.025
rand_iso        -0.459         0.132         0.134         0.003         0.010
combo_2         -0.265         0.150         0.151         0.005         0.015
combo_20        -0.234         0.127         0.127         0.006         0.013
```

## cos, trained, M = 3072

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate            -0.538         0.319         0.315         0.106         0.028
rand_cov         0.000         0.190         0.192         0.032         0.103
rand_rot        -0.500         0.167         0.174         0.024         0.043
combo_5         -0.105         0.124         0.131         0.037         0.080
data_pts         0.766         0.355         0.350         0.093         0.055
kmeans           1.000         0.366         0.359         0.125         0.065
rand_iso        -0.485         0.171         0.178         0.023         0.045
combo_2         -0.136         0.139         0.147         0.032         0.076
combo_20        -0.109         0.097         0.102         0.040         0.066
```

## cos, random, M = 32

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -1.798        -1.310         0.200         0.113         0.200         0.113         0.010         0.008         0.026         0.026
rand_cov         0.000         0.000         0.293         0.195         0.293         0.195         0.016         0.018         0.046         0.063
rand_rot        -1.812        -1.304         0.197         0.114         0.197         0.114         0.009         0.010         0.025         0.026
combo_5         -1.809        -1.311         0.201         0.112         0.201         0.112         0.010         0.008         0.025         0.025
data_pts         0.302         0.251         0.294         0.205         0.294         0.205         0.015         0.018         0.049         0.104
kmeans           1.000         1.000         0.354         0.253         0.354         0.253         0.020         0.022         0.100         0.169
rand_iso        -1.810        -1.313         0.195         0.112         0.195         0.112         0.010         0.009         0.024         0.026
combo_2         -1.814        -1.311         0.197         0.115         0.197         0.115         0.009         0.009         0.025         0.024
combo_20        -1.801        -1.306         0.204         0.115         0.204         0.115         0.010         0.009         0.025         0.024
```

## cos, random, M = 128

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -2.548        -1.643         0.255         0.154         0.255         0.154         0.007         0.006         0.033         0.043
rand_cov         0.000         0.000         0.348         0.243         0.348         0.243         0.011         0.013         0.061         0.100
rand_rot        -2.551        -1.637         0.251         0.158         0.251         0.158         0.007         0.006         0.035         0.046
combo_5         -2.550        -1.640         0.255         0.155         0.255         0.155         0.007         0.006         0.033         0.044
data_pts         0.322         0.305         0.361         0.257         0.361         0.257         0.012         0.014         0.090         0.166
kmeans           1.000         1.000         0.410         0.307         0.410         0.307         0.011         0.012         0.125         0.228
rand_iso        -2.541        -1.642         0.257         0.154         0.257         0.154         0.008         0.006         0.036         0.043
combo_2         -2.556        -1.641         0.253         0.156         0.253         0.156         0.007         0.006         0.034         0.044
combo_20        -2.548        -1.639         0.253         0.157         0.253         0.157         0.007         0.006         0.036         0.044
```

## cos, random, M = 512

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -3.109        -2.039         0.307         0.203         0.307         0.203         0.004         0.004         0.035         0.050
rand_cov         0.000         0.000         0.400         0.291         0.400         0.291         0.007         0.007         0.061         0.112
rand_rot        -3.103        -2.039         0.306         0.204         0.306         0.204         0.004         0.003         0.034         0.053
combo_5         -3.110        -2.043         0.308         0.201         0.308         0.202         0.004         0.004         0.035         0.052
data_pts         0.476         0.420         0.421         0.313         0.421         0.313         0.007         0.008         0.108         0.200
kmeans           1.000         1.000         0.454         0.351         0.454         0.351         0.005         0.006         0.119         0.234
rand_iso        -3.095        -2.038         0.306         0.205         0.306         0.205         0.004         0.004         0.035         0.053
combo_2         -3.109        -2.040         0.305         0.204         0.305         0.204         0.005         0.003         0.031         0.052
combo_20        -3.097        -2.042         0.307         0.204         0.307         0.204         0.005         0.003         0.034         0.053
```

## cos, random, M = 1536

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate            -3.536         0.366         0.366         0.002         0.031
rand_cov         0.000         0.454         0.454         0.004         0.056
rand_rot        -3.520         0.365         0.365         0.003         0.030
combo_5         -3.538         0.365         0.365         0.003         0.032
data_pts         0.616         0.477         0.477         0.004         0.096
kmeans           1.000         0.493         0.493         0.003         0.102
rand_iso        -3.534         0.365         0.365         0.003         0.031
combo_2         -3.545         0.364         0.364         0.003         0.028
combo_20        -3.531         0.365         0.365         0.003         0.029
```

## cos, random, M = 3072

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate            -2.602         0.320         0.320         0.002         0.043
rand_cov         0.000         0.388         0.388         0.004         0.101
rand_rot        -2.604         0.320         0.320         0.001         0.044
combo_5         -2.604         0.319         0.319         0.002         0.043
data_pts         0.647         0.413         0.413         0.003         0.193
kmeans           1.000         0.417         0.417         0.003         0.209
rand_iso        -2.611         0.318         0.318         0.001         0.042
combo_2         -2.609         0.319         0.319         0.002         0.044
combo_20        -2.612         0.317         0.317         0.002         0.043
```

## euclid, trained, M = 32

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.865         2.168         0.032         0.010         0.033         0.011         0.012         0.017         0.010         0.010
rand_cov         1.772         2.067         0.049         0.085         0.053         0.090         0.018         0.044         0.033         0.127
rand_rot         1.821         1.958         0.017         0.006         0.018         0.008         0.008         0.008         0.011         0.010
combo_5          1.266         1.517         0.005         0.008         0.006         0.008         0.002         0.009         0.003         0.013
data_pts         1.496         1.321         0.066         0.094         0.073         0.101         0.030         0.047         0.042         0.162
kmeans           0.880         0.864         0.114         0.133         0.123         0.138         0.057         0.145         0.059         0.187
rand_iso         1.940         2.345         0.025         0.033         0.026         0.034         0.009         0.045         0.015         0.040
combo_2          1.138         1.311         0.006         0.019         0.006         0.024         0.003         0.006         0.003         0.044
combo_20         1.492         1.803         0.009         0.017         0.009         0.017         0.004         0.008         0.006         0.025
```

## euclid, trained, M = 128

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.760         1.847         0.057         0.015         0.059         0.016         0.012         0.023         0.016         0.009
rand_cov         1.688         1.971         0.074         0.103         0.079         0.109         0.017         0.046         0.034         0.136
rand_rot         1.734         1.894         0.025         0.005         0.027         0.006         0.010         0.013         0.013         0.006
combo_5          1.198         1.357         0.012         0.028         0.013         0.034         0.006         0.007         0.005         0.044
data_pts         1.383         1.240         0.104         0.114         0.113         0.120         0.032         0.070         0.041         0.151
kmeans           0.836         0.835         0.163         0.167         0.173         0.172         0.039         0.163         0.055         0.166
rand_iso         1.879         2.280         0.041         0.046         0.043         0.047         0.011         0.047         0.017         0.040
combo_2          1.108         1.279         0.018         0.032         0.021         0.038         0.007         0.014         0.008         0.048
combo_20         1.467         1.654         0.018         0.013         0.021         0.014         0.008         0.011         0.007         0.026
```

## euclid, trained, M = 512

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.669         1.707         0.078         0.009         0.083         0.011         0.015         0.025         0.031         0.002
rand_cov         1.616         1.917         0.103         0.124         0.111         0.133         0.012         0.045         0.030         0.132
rand_rot         1.690         1.707         0.036         0.007         0.039         0.007         0.010         0.009         0.018         0.012
combo_5          1.137         1.305         0.020         0.040         0.022         0.050         0.008         0.011         0.008         0.066
data_pts         1.280         1.164         0.155         0.159         0.167         0.168         0.022         0.095         0.034         0.133
kmeans           0.809         0.803         0.207         0.217         0.218         0.220         0.023         0.150         0.036         0.136
rand_iso         1.823         2.228         0.063         0.071         0.066         0.075         0.009         0.040         0.017         0.051
combo_2          1.096         1.256         0.044         0.072         0.046         0.081         0.011         0.026         0.018         0.115
combo_20         1.378         1.551         0.028         0.019         0.031         0.024         0.010         0.007         0.010         0.037
```

## euclid, trained, M = 1536

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate             1.588         0.102         0.110         0.010         0.040
rand_cov         1.567         0.140         0.149         0.008         0.023
rand_rot         1.637         0.054         0.057         0.010         0.015
combo_5          1.113         0.033         0.037         0.012         0.014
data_pts         1.210         0.220         0.233         0.013         0.023
kmeans           0.803         0.243         0.256         0.016         0.026
rand_iso         1.782         0.094         0.097         0.006         0.016
combo_2          1.085         0.063         0.067         0.012         0.015
combo_20         1.323         0.034         0.037         0.013         0.016
```

## euclid, trained, M = 3072

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate             1.652         0.048         0.051         0.076         0.073
rand_cov         1.840         0.168         0.181         0.032         0.106
rand_rot         1.639         0.014         0.015         0.031         0.020
combo_5          1.251         0.073         0.083         0.020         0.111
data_pts         1.045         0.287         0.295         0.100         0.079
kmeans           0.772         0.311         0.311         0.133         0.086
rand_iso         2.180         0.120         0.131         0.032         0.051
combo_2          1.239         0.100         0.110         0.032         0.112
combo_20         1.474         0.024         0.027         0.019         0.037
```

## euclid, random, M = 32

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.742         1.833         0.159         0.093         0.159         0.093         0.008         0.008         0.018         0.020
rand_cov         1.175         1.378         0.288         0.191         0.288         0.191         0.015         0.016         0.051         0.069
rand_rot         1.754         1.822         0.168         0.096         0.168         0.096         0.008         0.009         0.017         0.020
combo_5          1.166         1.186         0.021         0.013         0.021         0.013         0.001         0.002         0.002         0.001
data_pts         1.065         1.283         0.294         0.205         0.294         0.205         0.016         0.019         0.048         0.104
kmeans           0.644         0.745         0.358         0.257         0.358         0.257         0.020         0.022         0.094         0.162
rand_iso         1.758         1.837         0.160         0.089         0.160         0.089         0.009         0.008         0.019         0.020
combo_2          1.026         1.034         0.028         0.004         0.028         0.004         0.002         0.001         0.003         0.001
combo_20         1.384         1.471         0.076         0.020         0.076         0.020         0.004         0.003         0.004         0.003
```

## euclid, random, M = 128

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.699         1.786         0.219         0.125         0.219         0.125         0.008         0.007         0.027         0.029
rand_cov         1.060         1.276         0.349         0.243         0.349         0.243         0.010         0.011         0.065         0.102
rand_rot         1.696         1.785         0.211         0.129         0.211         0.129         0.008         0.008         0.028         0.032
combo_5          1.091         1.090         0.065         0.014         0.065         0.014         0.004         0.001         0.004         0.004
data_pts         0.934         1.152         0.361         0.257         0.361         0.257         0.013         0.014         0.091         0.166
kmeans           0.593         0.696         0.413         0.311         0.413         0.311         0.011         0.012         0.118         0.217
rand_iso         1.693         1.790         0.217         0.120         0.217         0.120         0.008         0.007         0.029         0.031
combo_2          1.007         1.012         0.071         0.032         0.071         0.032         0.005         0.003         0.007         0.006
combo_20         1.357         1.351         0.082         0.034         0.082         0.034         0.005         0.004         0.010         0.005
```

## euclid, random, M = 512

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.641         1.752         0.249         0.163         0.249         0.163         0.007         0.005         0.026         0.039
rand_cov         0.986         1.201         0.401         0.294         0.401         0.294         0.005         0.005         0.058         0.113
rand_rot         1.642         1.745         0.262         0.163         0.262         0.163         0.007         0.006         0.028         0.039
combo_5          1.039         1.048         0.107         0.017         0.107         0.017         0.005         0.001         0.010         0.006
data_pts         0.825         1.041         0.421         0.313         0.421         0.313         0.007         0.008         0.108         0.199
kmeans           0.563         0.670         0.457         0.351         0.457         0.351         0.006         0.006         0.112         0.225
rand_iso         1.645         1.751         0.262         0.169         0.262         0.169         0.006         0.005         0.029         0.043
combo_2          0.997         1.003         0.164         0.068         0.164         0.068         0.008         0.007         0.019         0.015
combo_20         1.307         1.227         0.071         0.029         0.071         0.029         0.004         0.002         0.013         0.003
```

## euclid, random, M = 1536

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate             1.608         0.287         0.287         0.004         0.029
rand_cov         0.943         0.451         0.451         0.003         0.046
rand_rot         1.605         0.285         0.285         0.005         0.028
combo_5          1.019         0.129         0.129         0.006         0.011
data_pts         0.758         0.477         0.477         0.004         0.096
kmeans           0.551         0.487         0.487         0.003         0.096
rand_iso         1.614         0.299         0.299         0.005         0.031
combo_2          0.991         0.236         0.236         0.008         0.030
combo_20         1.231         0.113         0.113         0.004         0.012
```

## euclid, random, M = 3072

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate             1.709         0.229         0.229         0.004         0.046
rand_cov         1.129         0.384         0.384         0.002         0.087
rand_rot         1.707         0.227         0.227         0.004         0.047
combo_5          1.021         0.066         0.066         0.005         0.011
data_pts         0.926         0.413         0.413         0.003         0.192
kmeans           0.661         0.400         0.400         0.003         0.209
rand_iso         1.709         0.229         0.229         0.004         0.046
combo_2          0.999         0.134         0.134         0.009         0.033
combo_20         1.206         0.042         0.042         0.004         0.008
```
