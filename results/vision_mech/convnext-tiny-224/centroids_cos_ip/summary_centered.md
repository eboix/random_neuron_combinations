# Centroid test (centered MLP inputs)

Assignment `cos` = argmax cosine with unit centroids; `euclid` = nearest raw-scale centroid in raw (centered) x; `ip` = largest inner product with rows at native norm (largest pre-activation). `obj` = mean max-cosine of held-out positions to the set (cos only); `obj_norm` = 0 at the covariance-matched random null, 1 at spherical k-means; `dist` = mean squared distance to the nearest centroid / total variance (euclid only); `ve` / `ve_raw` = variance of the unit-normalized / raw x explained by the partition (1 - within-cluster SS / total SS); `nmi_*` = NMI of the partition with ImageNet class / spatial position, minus the NMI of a size-matched random partition.

## cos, trained, M = 32

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -0.222        -0.605         0.054         0.066         0.054         0.062         0.019         0.055         0.020         0.071
rand_cov          0.000         0.000         0.070         0.110         0.071         0.102         0.022         0.048         0.041         0.155
rand_rot         -0.424        -0.704         0.037         0.054         0.037         0.050         0.014         0.057         0.022         0.052
combo_5          -0.280        -0.591         0.050         0.078         0.050         0.073         0.016         0.050         0.020         0.080
data_pts          0.170         0.193         0.082         0.119         0.083         0.111         0.033         0.080         0.049         0.148
kmeans            1.000         1.000         0.119         0.151         0.120         0.141         0.060         0.174         0.058         0.171
rand_iso         -0.413        -0.696         0.036         0.052         0.036         0.048         0.014         0.058         0.021         0.048
rand_gcov        -0.317        -0.644         0.044         0.077         0.044         0.071         0.016         0.048         0.020         0.077
combo_2          -0.274        -0.562         0.049         0.092         0.049         0.085         0.016         0.046         0.020         0.090
combo_20         -0.289        -0.585         0.048         0.085         0.049         0.078         0.015         0.048         0.022         0.087
```

## cos, trained, M = 128

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -0.207        -0.645         0.096         0.090         0.097         0.085         0.016         0.074         0.021         0.061
rand_cov          0.000         0.000         0.100         0.129         0.101         0.120         0.015         0.041         0.041         0.153
rand_rot         -0.481        -0.824         0.057         0.080         0.058         0.074         0.008         0.039         0.022         0.063
combo_5          -0.301        -0.631         0.080         0.112         0.081         0.105         0.012         0.052         0.021         0.095
data_pts          0.263         0.230         0.126         0.153         0.127         0.143         0.026         0.101         0.046         0.131
kmeans            1.000         1.000         0.169         0.192         0.170         0.180         0.038         0.176         0.057         0.151
rand_iso         -0.468        -0.808         0.058         0.077         0.058         0.072         0.008         0.040         0.021         0.062
rand_gcov        -0.355        -0.738         0.070         0.100         0.070         0.093         0.009         0.035         0.021         0.081
combo_2          -0.283        -0.603         0.082         0.113         0.083         0.106         0.012         0.053         0.021         0.097
combo_20         -0.316        -0.668         0.077         0.109         0.078         0.102         0.011         0.043         0.021         0.092
```

## cos, trained, M = 512

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -0.179        -0.509         0.147         0.144         0.149         0.138         0.010         0.109         0.015         0.044
rand_cov          0.000         0.000         0.137         0.156         0.138         0.147         0.007         0.036         0.027         0.133
rand_rot         -0.533        -0.777         0.091         0.110         0.092         0.104         0.004         0.023         0.012         0.056
combo_5          -0.313        -0.536         0.123         0.148         0.124         0.141         0.007         0.064         0.013         0.079
data_pts          0.387         0.377         0.184         0.211         0.186         0.198         0.014         0.103         0.032         0.100
kmeans            1.000         1.000         0.220         0.252         0.222         0.237         0.020         0.159         0.034         0.106
rand_iso         -0.518        -0.756         0.092         0.111         0.093         0.105         0.003         0.023         0.013         0.056
rand_gcov        -0.397        -0.690         0.106         0.129         0.107         0.121         0.004         0.023         0.012         0.072
combo_2          -0.280        -0.519         0.129         0.151         0.130         0.144         0.008         0.071         0.020         0.077
combo_20         -0.348        -0.606         0.116         0.141         0.118         0.134         0.005         0.042         0.014         0.078
```

## cos, trained, M = 1536

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate             -0.165         0.213         0.215         0.005         0.010
rand_cov          0.000         0.191         0.193         0.004         0.016
rand_rot         -0.605         0.145         0.146         0.002         0.007
combo_5          -0.338         0.181         0.183         0.004         0.009
data_pts          0.509         0.252         0.254         0.008         0.019
kmeans            1.000         0.274         0.276         0.012         0.021
rand_iso         -0.586         0.146         0.147         0.002         0.007
rand_gcov        -0.448         0.161         0.162         0.002         0.007
combo_2          -0.300         0.185         0.188         0.005         0.012
combo_20         -0.388         0.172         0.174         0.003         0.007
```

## cos, trained, M = 3072

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate             -0.282         0.307         0.305         0.167         0.022
rand_cov          0.000         0.238         0.231         0.024         0.098
rand_rot         -0.773         0.227         0.221         0.011         0.032
combo_5          -0.460         0.275         0.275         0.095         0.052
data_pts          0.609         0.370         0.356         0.091         0.046
kmeans            1.000         0.387         0.372         0.121         0.056
rand_iso         -0.751         0.227         0.221         0.011         0.032
rand_gcov        -0.682         0.235         0.227         0.013         0.045
combo_2          -0.410         0.275         0.275         0.114         0.050
combo_20         -0.564         0.259         0.255         0.049         0.051
```

## cos, random, M = 32

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -1.777        -1.307         0.200         0.113         0.201         0.114         0.010         0.008         0.026         0.026
rand_cov          0.000         0.000         0.292         0.195         0.292         0.196         0.015         0.017         0.046         0.064
rand_rot         -1.790        -1.303         0.196         0.114         0.197         0.114         0.009         0.010         0.025         0.027
combo_5          -1.789        -1.309         0.201         0.112         0.202         0.112         0.010         0.008         0.025         0.025
data_pts          0.302         0.242         0.293         0.205         0.293         0.205         0.015         0.018         0.048         0.103
kmeans            1.000         1.000         0.354         0.252         0.354         0.253         0.020         0.022         0.099         0.169
rand_iso         -1.790        -1.311         0.194         0.111         0.195         0.112         0.010         0.009         0.024         0.026
rand_gcov        -1.776        -1.302         0.201         0.117         0.202         0.117         0.010         0.010         0.024         0.025
combo_2          -1.789        -1.310         0.196         0.115         0.197         0.116         0.009         0.010         0.024         0.025
combo_20         -1.778        -1.306         0.203         0.114         0.204         0.114         0.010         0.009         0.025         0.024
```

## cos, random, M = 128

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -2.544        -1.634         0.254         0.153         0.255         0.153         0.007         0.006         0.031         0.043
rand_cov          0.000         0.000         0.347         0.243         0.348         0.243         0.011         0.013         0.062         0.101
rand_rot         -2.545        -1.629         0.251         0.158         0.252         0.158         0.007         0.006         0.034         0.047
combo_5          -2.545        -1.633         0.254         0.155         0.255         0.155         0.007         0.006         0.033         0.045
data_pts          0.318         0.301         0.360         0.257         0.361         0.258         0.013         0.014         0.090         0.166
kmeans            1.000         1.000         0.409         0.307         0.409         0.307         0.011         0.011         0.125         0.228
rand_iso         -2.533        -1.635         0.257         0.154         0.258         0.154         0.008         0.005         0.035         0.043
rand_gcov        -2.540        -1.630         0.254         0.157         0.254         0.157         0.008         0.005         0.033         0.043
combo_2          -2.551        -1.633         0.252         0.156         0.253         0.156         0.007         0.006         0.034         0.044
combo_20         -2.541        -1.631         0.253         0.156         0.253         0.157         0.006         0.006         0.035         0.045
```

## cos, random, M = 512

```
               obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate             -3.102        -2.024         0.306         0.202         0.306         0.202         0.004         0.004         0.035         0.050
rand_cov          0.000         0.000         0.400         0.292         0.400         0.292         0.008         0.007         0.061         0.113
rand_rot         -3.098        -2.023         0.305         0.204         0.306         0.205         0.004         0.003         0.034         0.054
combo_5          -3.103        -2.028         0.308         0.201         0.308         0.202         0.004         0.004         0.035         0.052
data_pts          0.471         0.417         0.421         0.313         0.422         0.314         0.007         0.008         0.108         0.200
kmeans            1.000         1.000         0.453         0.351         0.454         0.351         0.005         0.006         0.118         0.235
rand_iso         -3.092        -2.023         0.307         0.204         0.307         0.205         0.005         0.004         0.034         0.054
rand_gcov        -3.101        -2.022         0.307         0.204         0.307         0.204         0.004         0.003         0.035         0.050
combo_2          -3.103        -2.023         0.305         0.204         0.305         0.204         0.005         0.003         0.032         0.053
combo_20         -3.090        -2.025         0.306         0.204         0.307         0.205         0.005         0.003         0.035         0.052
```

## cos, random, M = 1536

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate             -3.526         0.366         0.366         0.003         0.031
rand_cov          0.000         0.455         0.455         0.004         0.056
rand_rot         -3.510         0.365         0.366         0.003         0.030
combo_5          -3.529         0.365         0.365         0.003         0.031
data_pts          0.616         0.477         0.477         0.004         0.095
kmeans            1.000         0.492         0.493         0.003         0.101
rand_iso         -3.527         0.364         0.364         0.003         0.032
rand_gcov        -3.526         0.365         0.365         0.003         0.032
combo_2          -3.535         0.363         0.363         0.003         0.030
combo_20         -3.521         0.363         0.364         0.003         0.029
```

## cos, random, M = 3072

```
               obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate             -2.586         0.320         0.320         0.002         0.043
rand_cov          0.000         0.389         0.389         0.003         0.101
rand_rot         -2.589         0.320         0.320         0.002         0.043
combo_5          -2.588         0.319         0.320         0.002         0.043
data_pts          0.646         0.413         0.413         0.003         0.192
kmeans            1.000         0.417         0.417         0.002         0.207
rand_iso         -2.595         0.318         0.318         0.001         0.043
rand_gcov        -2.594         0.319         0.320         0.001         0.044
combo_2          -2.593         0.319         0.319         0.002         0.042
combo_20         -2.596         0.317         0.317         0.002         0.043
```

## ip, trained, M = 32

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.054         0.061         0.054         0.058        30.976        26.723         0.019         0.059         0.021         0.063
rand_rot          0.036         0.053         0.037         0.049        30.341        29.459         0.015         0.057         0.021         0.050
combo_5           0.049         0.078         0.049         0.074        28.811        26.941         0.015         0.049         0.020         0.079
rand_iso          0.036         0.053         0.036         0.049        30.008        29.758         0.014         0.057         0.020         0.050
rand_gcov         0.044         0.077         0.044         0.071        30.422        27.877         0.016         0.047         0.021         0.078
combo_2           0.048         0.090         0.048         0.083        28.719        23.471         0.015         0.046         0.020         0.085
combo_20          0.047         0.083         0.048         0.077        29.425        25.849         0.015         0.048         0.022         0.085
```

## ip, trained, M = 128

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.094         0.085         0.096         0.080       113.047       102.075         0.016         0.076         0.023         0.056
rand_rot          0.055         0.079         0.056         0.073       106.201       105.881         0.009         0.041         0.020         0.062
combo_5           0.078         0.111         0.078         0.104       101.541        66.595         0.012         0.052         0.020         0.096
rand_iso          0.056         0.076         0.057         0.070       107.177       105.762         0.008         0.042         0.021         0.061
rand_gcov         0.069         0.100         0.069         0.093       110.992        89.166         0.010         0.035         0.021         0.081
combo_2           0.079         0.112         0.080         0.104        93.979        57.105         0.012         0.053         0.020         0.098
combo_20          0.075         0.108         0.076         0.101       100.419        78.098         0.011         0.043         0.021         0.091
```

## ip, trained, M = 512

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.146         0.139         0.148         0.132       385.440       402.631         0.011         0.106         0.017         0.038
rand_rot          0.088         0.108         0.090         0.102       354.832       352.283         0.005         0.025         0.013         0.056
combo_5           0.116         0.146         0.118         0.138       306.410       186.579         0.008         0.063         0.015         0.081
rand_iso          0.088         0.109         0.089         0.103       359.108       361.188         0.004         0.026         0.014         0.057
rand_gcov         0.104         0.129         0.105         0.121       392.963       264.421         0.005         0.023         0.013         0.073
combo_2           0.123         0.149         0.125         0.142       282.173       192.732         0.009         0.070         0.020         0.078
combo_20          0.110         0.139         0.111         0.132       325.773       244.097         0.007         0.043         0.015         0.078
```

## ip, trained, M = 1536

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate              0.209         0.211       964.239         0.008         0.013
rand_rot          0.137         0.138       887.755         0.003         0.009
combo_5           0.167         0.169       687.724         0.005         0.010
rand_iso          0.137         0.139       804.289         0.003         0.010
rand_gcov         0.157         0.158       980.677         0.003         0.008
combo_2           0.171         0.174       602.094         0.007         0.014
combo_20          0.160         0.161       741.043         0.004         0.010
```

## ip, trained, M = 3072

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate              0.303         0.299      1718.519         0.157         0.017
rand_rot          0.211         0.205      1550.656         0.013         0.035
combo_5           0.262         0.260       717.610         0.093         0.054
rand_iso          0.210         0.204      1545.365         0.013         0.035
rand_gcov         0.230         0.223      1052.886         0.013         0.047
combo_2           0.263         0.262       725.864         0.112         0.052
combo_20          0.242         0.237       869.593         0.047         0.055
```

## ip, random, M = 32

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.200         0.113         0.200         0.113        27.052        29.235         0.009         0.008         0.026         0.026
rand_rot          0.196         0.114         0.197         0.114        27.013        29.311         0.010         0.009         0.025         0.026
combo_5           0.200         0.112         0.201         0.112        26.559        28.520         0.010         0.008         0.026         0.024
rand_iso          0.193         0.111         0.193         0.111        27.000        29.028         0.010         0.008         0.024         0.026
rand_gcov         0.201         0.117         0.202         0.118        26.335        28.256         0.010         0.009         0.024         0.024
combo_2           0.196         0.115         0.197         0.116        27.597        29.391         0.009         0.009         0.025         0.025
combo_20          0.202         0.114         0.203         0.114        25.661        28.668         0.010         0.008         0.024         0.024
```

## ip, random, M = 128

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.253         0.152         0.254         0.153        85.430       103.413         0.007         0.006         0.033         0.043
rand_rot          0.249         0.158         0.250         0.158        83.305       101.602         0.007         0.006         0.035         0.046
combo_5           0.254         0.155         0.254         0.155        82.432       101.095         0.008         0.006         0.032         0.044
rand_iso          0.256         0.154         0.256         0.154        82.339       102.668         0.008         0.006         0.035         0.043
rand_gcov         0.252         0.157         0.253         0.157        81.980       100.370         0.008         0.006         0.033         0.043
combo_2           0.251         0.156         0.252         0.156        86.054       102.162         0.007         0.006         0.035         0.044
combo_20          0.252         0.156         0.252         0.157        86.083       101.350         0.007         0.006         0.035         0.044
```

## ip, random, M = 512

```
                     ve                      ve_raw                       eff_k                     nmi_class                    nmi_pos              
layer     stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                  
gate              0.304         0.202         0.305         0.203       242.226       318.147         0.004         0.004         0.035         0.049
rand_rot          0.304         0.204         0.304         0.205       244.587       335.251         0.005         0.004         0.034         0.053
combo_5           0.305         0.200         0.306         0.201       255.575       342.869         0.004         0.004         0.035         0.051
rand_iso          0.305         0.204         0.305         0.204       249.156       322.040         0.004         0.004         0.033         0.054
rand_gcov         0.304         0.203         0.305         0.204       251.156       318.131         0.004         0.004         0.033         0.051
combo_2           0.303         0.203         0.303         0.203       239.923       321.637         0.004         0.004         0.032         0.051
combo_20          0.305         0.203         0.305         0.204       244.553       327.814         0.004         0.004         0.033         0.052
```

## ip, random, M = 1536

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                            
gate              0.362         0.362       574.389         0.003         0.031
rand_rot          0.362         0.362       534.947         0.004         0.030
combo_5           0.362         0.362       570.798         0.003         0.031
rand_iso          0.360         0.361       520.970         0.003         0.032
rand_gcov         0.361         0.361       559.301         0.003         0.031
combo_2           0.360         0.360       564.657         0.003         0.031
combo_20          0.361         0.361       551.973         0.003         0.030
```

## ip, random, M = 3072

```
                     ve        ve_raw         eff_k       nmi_class      nmi_pos
layer     stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                            
gate              0.316         0.316      1231.502         0.002         0.044
rand_rot          0.317         0.317      1397.230         0.001         0.044
combo_5           0.316         0.316      1357.621         0.002         0.043
rand_iso          0.315         0.316      1448.936         0.001         0.043
rand_gcov         0.315         0.316      1364.443         0.002         0.044
combo_2           0.315         0.316      1316.504         0.002         0.042
combo_20          0.315         0.315      1405.200         0.001         0.044
```
