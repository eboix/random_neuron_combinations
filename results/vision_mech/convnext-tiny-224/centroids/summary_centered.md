# Centroid test (centered MLP inputs)

Assignment `cos` = argmax cosine with unit centroids; `euclid` = nearest raw-scale centroid in raw (centered) x. `obj` = mean max-cosine of held-out positions to the set (cos only); `obj_norm` = 0 at the covariance-matched random null, 1 at spherical k-means; `dist` = mean squared distance to the nearest centroid / total variance (euclid only); `ve` / `ve_raw` = variance of the unit-normalized / raw x explained by the partition (1 - within-cluster SS / total SS); `nmi_*` = NMI of the partition with ImageNet class / spatial position, minus the NMI of a size-matched random partition.

## cos, trained, M = 32

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -0.222        -0.605         0.054         0.066         0.054         0.062         0.019         0.055         0.020         0.071
rand_cov         0.000         0.000         0.070         0.110         0.071         0.102         0.022         0.048         0.041         0.155
rand_rot        -0.424        -0.704         0.037         0.054         0.037         0.050         0.014         0.057         0.022         0.052
combo_5         -0.280        -0.591         0.050         0.078         0.050         0.073         0.016         0.050         0.020         0.080
data_pts         0.170         0.193         0.082         0.119         0.083         0.111         0.033         0.080         0.049         0.148
kmeans           1.000         1.000         0.119         0.151         0.120         0.141         0.060         0.174         0.058         0.171
rand_iso        -0.413        -0.696         0.036         0.052         0.036         0.048         0.014         0.058         0.021         0.048
combo_2         -0.274        -0.562         0.049         0.092         0.049         0.085         0.016         0.046         0.020         0.090
combo_20        -0.289        -0.585         0.048         0.085         0.049         0.078         0.015         0.048         0.022         0.087
```

## cos, trained, M = 128

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -0.207        -0.645         0.096         0.090         0.097         0.085         0.016         0.074         0.021         0.061
rand_cov         0.000         0.000         0.100         0.129         0.101         0.120         0.015         0.041         0.041         0.153
rand_rot        -0.481        -0.824         0.057         0.080         0.058         0.074         0.008         0.039         0.022         0.063
combo_5         -0.301        -0.631         0.080         0.112         0.081         0.105         0.012         0.052         0.021         0.095
data_pts         0.263         0.230         0.126         0.153         0.127         0.143         0.026         0.101         0.046         0.131
kmeans           1.000         1.000         0.169         0.192         0.170         0.180         0.038         0.176         0.057         0.151
rand_iso        -0.468        -0.808         0.058         0.077         0.058         0.072         0.008         0.040         0.021         0.062
combo_2         -0.283        -0.603         0.082         0.113         0.083         0.106         0.012         0.053         0.021         0.097
combo_20        -0.316        -0.668         0.077         0.109         0.078         0.102         0.011         0.043         0.021         0.092
```

## cos, trained, M = 512

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -0.179        -0.509         0.147         0.144         0.149         0.138         0.010         0.109         0.015         0.044
rand_cov         0.000         0.000         0.137         0.156         0.138         0.147         0.007         0.036         0.027         0.133
rand_rot        -0.533        -0.777         0.091         0.110         0.092         0.104         0.004         0.023         0.012         0.056
combo_5         -0.313        -0.536         0.123         0.148         0.124         0.141         0.007         0.064         0.013         0.079
data_pts         0.387         0.377         0.184         0.211         0.186         0.198         0.014         0.103         0.032         0.100
kmeans           1.000         1.000         0.220         0.252         0.222         0.237         0.020         0.159         0.034         0.106
rand_iso        -0.518        -0.756         0.092         0.111         0.093         0.105         0.003         0.023         0.013         0.056
combo_2         -0.280        -0.519         0.129         0.151         0.130         0.144         0.008         0.071         0.020         0.077
combo_20        -0.348        -0.606         0.116         0.141         0.118         0.134         0.005         0.042         0.014         0.078
```

## cos, trained, M = 1536

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate            -0.165         0.213         0.215         0.005         0.010
rand_cov         0.000         0.191         0.193         0.004         0.016
rand_rot        -0.605         0.145         0.146         0.002         0.007
combo_5         -0.338         0.181         0.183         0.004         0.009
data_pts         0.509         0.252         0.254         0.008         0.019
kmeans           1.000         0.274         0.276         0.012         0.021
rand_iso        -0.586         0.146         0.147         0.002         0.007
combo_2         -0.300         0.185         0.188         0.005         0.012
combo_20        -0.388         0.172         0.174         0.003         0.007
```

## cos, trained, M = 3072

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate            -0.282         0.307         0.305         0.167         0.022
rand_cov         0.000         0.238         0.231         0.024         0.098
rand_rot        -0.773         0.227         0.221         0.011         0.032
combo_5         -0.460         0.275         0.275         0.095         0.052
data_pts         0.609         0.370         0.356         0.091         0.046
kmeans           1.000         0.387         0.372         0.121         0.056
rand_iso        -0.751         0.227         0.221         0.011         0.032
combo_2         -0.410         0.275         0.275         0.114         0.050
combo_20        -0.564         0.259         0.255         0.049         0.051
```

## cos, random, M = 32

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -1.777        -1.307         0.200         0.113         0.201         0.114         0.010         0.008         0.026         0.026
rand_cov         0.000         0.000         0.292         0.195         0.292         0.196         0.015         0.017         0.046         0.064
rand_rot        -1.790        -1.303         0.196         0.114         0.197         0.114         0.009         0.010         0.025         0.027
combo_5         -1.789        -1.309         0.201         0.112         0.202         0.112         0.010         0.008         0.025         0.025
data_pts         0.302         0.242         0.293         0.205         0.293         0.205         0.015         0.018         0.048         0.103
kmeans           1.000         1.000         0.354         0.252         0.354         0.253         0.020         0.022         0.099         0.169
rand_iso        -1.790        -1.311         0.194         0.111         0.195         0.112         0.010         0.009         0.024         0.026
combo_2         -1.789        -1.310         0.196         0.115         0.197         0.116         0.009         0.010         0.024         0.025
combo_20        -1.778        -1.306         0.203         0.114         0.204         0.114         0.010         0.009         0.025         0.024
```

## cos, random, M = 128

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -2.544        -1.634         0.254         0.153         0.255         0.153         0.007         0.006         0.031         0.043
rand_cov         0.000         0.000         0.347         0.243         0.348         0.243         0.011         0.013         0.062         0.101
rand_rot        -2.545        -1.629         0.251         0.158         0.252         0.158         0.007         0.006         0.034         0.047
combo_5         -2.545        -1.633         0.254         0.155         0.255         0.155         0.007         0.006         0.033         0.045
data_pts         0.318         0.301         0.360         0.257         0.361         0.258         0.013         0.014         0.090         0.166
kmeans           1.000         1.000         0.409         0.307         0.409         0.307         0.011         0.011         0.125         0.228
rand_iso        -2.533        -1.635         0.257         0.154         0.258         0.154         0.008         0.005         0.035         0.043
combo_2         -2.551        -1.633         0.252         0.156         0.253         0.156         0.007         0.006         0.034         0.044
combo_20        -2.541        -1.631         0.253         0.156         0.253         0.157         0.006         0.006         0.035         0.045
```

## cos, random, M = 512

```
              obj_norm                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate            -3.102        -2.024         0.306         0.202         0.306         0.202         0.004         0.004         0.035         0.050
rand_cov         0.000         0.000         0.400         0.292         0.400         0.292         0.008         0.007         0.061         0.113
rand_rot        -3.098        -2.023         0.305         0.204         0.306         0.205         0.004         0.003         0.034         0.054
combo_5         -3.103        -2.028         0.308         0.201         0.308         0.202         0.004         0.004         0.035         0.052
data_pts         0.471         0.417         0.421         0.313         0.422         0.314         0.007         0.008         0.108         0.200
kmeans           1.000         1.000         0.453         0.351         0.454         0.351         0.005         0.006         0.118         0.235
rand_iso        -3.092        -2.023         0.307         0.204         0.307         0.205         0.005         0.004         0.034         0.054
combo_2         -3.103        -2.023         0.305         0.204         0.305         0.204         0.005         0.003         0.032         0.053
combo_20        -3.090        -2.025         0.306         0.204         0.307         0.205         0.005         0.003         0.035         0.052
```

## cos, random, M = 1536

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate            -3.526         0.366         0.366         0.003         0.031
rand_cov         0.000         0.455         0.455         0.004         0.056
rand_rot        -3.510         0.365         0.366         0.003         0.030
combo_5         -3.529         0.365         0.365         0.003         0.031
data_pts         0.616         0.477         0.477         0.004         0.095
kmeans           1.000         0.492         0.493         0.003         0.101
rand_iso        -3.527         0.364         0.364         0.003         0.032
combo_2         -3.535         0.363         0.363         0.003         0.030
combo_20        -3.521         0.363         0.364         0.003         0.029
```

## cos, random, M = 3072

```
              obj_norm            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate            -2.586         0.320         0.320         0.002         0.043
rand_cov         0.000         0.389         0.389         0.003         0.101
rand_rot        -2.589         0.320         0.320         0.002         0.043
combo_5         -2.588         0.319         0.320         0.002         0.043
data_pts         0.646         0.413         0.413         0.003         0.192
kmeans           1.000         0.417         0.417         0.002         0.207
rand_iso        -2.595         0.318         0.318         0.001         0.043
combo_2         -2.593         0.319         0.319         0.002         0.042
combo_20        -2.596         0.317         0.317         0.002         0.043
```

## euclid, trained, M = 32

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.601         1.548         0.035         0.039         0.036         0.040         0.013         0.018         0.010         0.061
rand_cov         1.609         1.605         0.058         0.108         0.061         0.102         0.019         0.050         0.037         0.152
rand_rot         1.635         1.495         0.018         0.007         0.019         0.008         0.009         0.005         0.013         0.010
combo_5          1.156         1.175         0.007         0.009         0.007         0.010         0.003         0.009         0.003         0.010
data_pts         1.496         1.321         0.070         0.105         0.073         0.101         0.030         0.047         0.042         0.162
kmeans           0.880         0.864         0.120         0.145         0.123         0.138         0.058         0.146         0.059         0.187
rand_iso         1.745         1.796         0.029         0.044         0.030         0.042         0.011         0.050         0.017         0.046
combo_2          1.020         1.025         0.009         0.007         0.009         0.008         0.005         0.008         0.004         0.013
combo_20         1.381         1.420         0.013         0.012         0.014         0.013         0.006         0.013         0.006         0.018
```

## euclid, trained, M = 128

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.514         1.386         0.066         0.027         0.068         0.029         0.016         0.036         0.015         0.031
rand_cov         1.523         1.536         0.086         0.126         0.088         0.121         0.016         0.048         0.036         0.145
rand_rot         1.557         1.446         0.028         0.005         0.029         0.007         0.011         0.014         0.016         0.006
combo_5          1.073         1.083         0.014         0.019         0.014         0.018         0.006         0.013         0.006         0.027
data_pts         1.383         1.240         0.109         0.125         0.113         0.120         0.032         0.070         0.041         0.151
kmeans           0.836         0.835         0.168         0.179         0.172         0.172         0.039         0.163         0.055         0.166
rand_iso         1.682         1.747         0.045         0.062         0.046         0.060         0.011         0.048         0.019         0.050
combo_2          0.998         1.002         0.021         0.041         0.022         0.041         0.007         0.018         0.008         0.048
combo_20         1.326         1.334         0.023         0.022         0.024         0.024         0.007         0.016         0.008         0.029
```

## euclid, trained, M = 512

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.436         1.298         0.101         0.041         0.105         0.040         0.015         0.066         0.017         0.026
rand_cov         1.459         1.495         0.117         0.153         0.122         0.149         0.010         0.035         0.027         0.122
rand_rot         1.509         1.332         0.043         0.018         0.044         0.018         0.011         0.011         0.018         0.032
combo_5          1.032         1.038         0.027         0.025         0.028         0.024         0.012         0.018         0.012         0.028
data_pts         1.280         1.164         0.161         0.171         0.167         0.168         0.022         0.095         0.034         0.133
kmeans           0.809         0.803         0.212         0.228         0.218         0.220         0.023         0.150         0.036         0.136
rand_iso         1.635         1.711         0.072         0.095         0.074         0.094         0.008         0.038         0.017         0.059
combo_2          0.985         0.993         0.058         0.080         0.060         0.081         0.013         0.039         0.021         0.085
combo_20         1.257         1.233         0.040         0.014         0.042         0.016         0.010         0.019         0.012         0.019
```

## euclid, trained, M = 1536

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate             1.386         0.142         0.149         0.011         0.016
rand_cov         1.414         0.162         0.167         0.007         0.019
rand_rot         1.475         0.061         0.064         0.010         0.017
combo_5          1.009         0.042         0.044         0.007         0.013
data_pts         1.210         0.224         0.233         0.013         0.023
kmeans           0.803         0.249         0.256         0.016         0.026
rand_iso         1.598         0.108         0.112         0.005         0.013
combo_2          0.976         0.087         0.091         0.015         0.019
combo_20         1.198         0.043         0.045         0.011         0.012
```

## euclid, trained, M = 3072

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate             1.228         0.095         0.103         0.102         0.075
rand_cov         1.440         0.218         0.223         0.021         0.086
rand_rot         1.257         0.018         0.018         0.031         0.019
combo_5          1.011         0.076         0.078         0.032         0.072
data_pts         1.045         0.294         0.295         0.100         0.079
kmeans           0.772         0.319         0.311         0.133         0.086
rand_iso         1.671         0.155         0.162         0.025         0.051
combo_2          0.981         0.138         0.140         0.061         0.098
combo_20         1.168         0.033         0.039         0.015         0.037
```

## euclid, random, M = 32

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.737         1.825         0.159         0.093         0.159         0.094         0.007         0.009         0.018         0.020
rand_cov         1.170         1.375         0.287         0.191         0.287         0.192         0.015         0.016         0.051         0.070
rand_rot         1.747         1.816         0.169         0.095         0.169         0.095         0.008         0.008         0.017         0.020
combo_5          1.163         1.183         0.021         0.014         0.021         0.014         0.001         0.002         0.002         0.001
data_pts         1.065         1.283         0.294         0.205         0.294         0.205         0.016         0.019         0.048         0.104
kmeans           0.644         0.745         0.358         0.256         0.358         0.257         0.020         0.022         0.094         0.163
rand_iso         1.751         1.830         0.160         0.090         0.161         0.090         0.009         0.007         0.020         0.020
combo_2          1.021         1.030         0.027         0.004         0.027         0.004         0.002         0.000         0.003         0.002
combo_20         1.376         1.465         0.075         0.020         0.074         0.019         0.004         0.002         0.004         0.003
```

## euclid, random, M = 128

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.693         1.777         0.218         0.125         0.219         0.125         0.008         0.007         0.028         0.030
rand_cov         1.056         1.273         0.348         0.243         0.349         0.243         0.010         0.011         0.065         0.102
rand_rot         1.690         1.777         0.210         0.129         0.210         0.129         0.008         0.008         0.028         0.033
combo_5          1.087         1.086         0.067         0.013         0.067         0.013         0.004         0.002         0.004         0.004
data_pts         0.934         1.152         0.360         0.257         0.361         0.257         0.013         0.014         0.091         0.166
kmeans           0.593         0.696         0.413         0.310         0.413         0.311         0.011         0.012         0.118         0.217
rand_iso         1.687         1.783         0.218         0.120         0.218         0.121         0.008         0.007         0.028         0.032
combo_2          1.003         1.007         0.076         0.031         0.077         0.031         0.004         0.003         0.007         0.006
combo_20         1.352         1.348         0.082         0.035         0.083         0.036         0.005         0.004         0.010         0.005
```

## euclid, random, M = 512

```
                  dist                          ve                      ve_raw                     nmi_class                    nmi_pos              
layer    stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2 stage2.block8 stage3.block2
set                                                                                                                                                 
gate             1.635         1.744         0.249         0.163         0.250         0.163         0.007         0.006         0.027         0.039
rand_cov         0.983         1.198         0.400         0.294         0.401         0.294         0.005         0.005         0.058         0.114
rand_rot         1.636         1.738         0.262         0.163         0.262         0.163         0.006         0.006         0.028         0.038
combo_5          1.035         1.044         0.108         0.017         0.108         0.017         0.006         0.001         0.010         0.006
data_pts         0.825         1.041         0.420         0.312         0.421         0.313         0.007         0.008         0.108         0.199
kmeans           0.563         0.670         0.456         0.351         0.457         0.351         0.006         0.006         0.112         0.225
rand_iso         1.639         1.743         0.261         0.169         0.261         0.170         0.006         0.005         0.028         0.043
combo_2          0.993         0.999         0.164         0.066         0.164         0.067         0.008         0.006         0.019         0.014
combo_20         1.302         1.222         0.076         0.029         0.077         0.028         0.004         0.003         0.014         0.003
```

## euclid, random, M = 1536

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage2.block8 stage2.block8 stage2.block8 stage2.block8 stage2.block8
set                                                                           
gate             1.601         0.286         0.287         0.004         0.029
rand_cov         0.941         0.451         0.452         0.003         0.046
rand_rot         1.599         0.286         0.286         0.005         0.027
combo_5          1.015         0.129         0.129         0.006         0.011
data_pts         0.758         0.476         0.477         0.004         0.096
kmeans           0.551         0.486         0.487         0.003         0.096
rand_iso         1.608         0.299         0.300         0.006         0.031
combo_2          0.987         0.235         0.236         0.009         0.029
combo_20         1.225         0.111         0.111         0.004         0.012
```

## euclid, random, M = 3072

```
                  dist            ve        ve_raw       nmi_class      nmi_pos
layer    stage3.block2 stage3.block2 stage3.block2 stage3.block2 stage3.block2
set                                                                           
gate             1.702         0.229         0.229         0.004         0.046
rand_cov         1.126         0.384         0.385         0.002         0.087
rand_rot         1.699         0.226         0.227         0.004         0.047
combo_5          1.017         0.069         0.070         0.006         0.010
data_pts         0.926         0.412         0.413         0.003         0.192
kmeans           0.661         0.400         0.400         0.003         0.209
rand_iso         1.701         0.230         0.230         0.004         0.046
combo_2          0.994         0.133         0.134         0.008         0.034
combo_20         1.198         0.038         0.038         0.004         0.008
```
