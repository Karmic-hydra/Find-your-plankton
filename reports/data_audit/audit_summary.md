# WHOI 2014 Local Dataset Audit

- Generated (UTC): 2026-03-19T17:08:23.819077+00:00
- Dataset root: data/2014
- Local classes: 94
- Local images: 329832

## Reference Gap
- Expected classes: 103
- Expected images: 3500000
- Class gap: -9
- Image gap: -3170168

## Class Distribution
- Min per class: 1
- Q1 per class: 8.00
- Median per class: 27.50
- Mean per class: 3508.85
- Q3 per class: 173.25
- Max per class: 266156
- Std per class: 27495.87
- Classes with <10 images: 29
- Classes with <50 images: 54
- Classes with <100 images: 66
- Classes with <500 images: 80

## Top 10 Largest Classes
| Class | Images |
|---|---:|
| mix | 266156 |
| detritus | 36346 |
| Leptocylindrus | 4246 |
| dino30 | 3816 |
| mix_elongated | 3528 |
| Cylindrotheca | 2345 |
| Rhizosolenia | 2199 |
| Chaetoceros | 1871 |
| Ciliate_mix | 1074 |
| Guinardia_delicatula | 949 |

## Top 10 Smallest Classes
| Class | Images |
|---|---:|
| Akashiwo | 1 |
| Chaetoceros_didymus_flagellate | 1 |
| G_delicatula_detritus | 1 |
| Lauderia | 1 |
| Strombidium_capitatum | 1 |
| Strombidium_conicum | 2 |
| Euplotes_sp | 3 |
| Leegaardiella_ovalis | 3 |
| Odontella | 3 |
| Protoperidinium | 3 |

## File Extensions
| Extension | Count |
|---|---:|
| .png | 329832 |

## Phase-2 Recommendation
- Freeze class policy before generating splits.
- Exclude classes under minimum-count threshold and log rationale.
- Use stratified manifests shared by CNN and traditional ML pipelines.
