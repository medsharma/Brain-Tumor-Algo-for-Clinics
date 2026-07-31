# The out-of-scope image set

Owned by session B. This is the set used to test whether the classifier will
refuse an image it should not judge.

The images themselves are not committed. Some are large, some are redistributions
of third-party data, and all of them are reproducible from this file plus the
build script. They are listed in `.gitignore`. The manifest,
`data/out_of_scope/manifest.csv`, **is** committed, so every image is accounted
for by path, category, source and licence.

## Rebuild it

```bash
python -m pip install pyarrow medmnist pydicom
python analysis/out_of_scope_rejection.py build
```

That regenerates categories 1 and 3 from scratch and copies categories 2 and 4
out of `data/out_of_scope/_raw/`. It needs the downloads below to be present
first. The download steps are in `Acquisition` at the bottom.

Everything is seeded (`SEED = 20260730` in the build script), so a rebuild is
byte-identical for the generated categories and file-identical for the copied
ones.

## What is in it

3,192 images in four categories.

| category | n | what it is |
|---|---|---|
| 1, corrupted scans | 1,560 | brain MRI from the internal train split, damaged in software |
| 2, non-brain medical | 1,032 | other body parts, other machines |
| 3, non-medical | 480 | photographs, documents, chart screenshots |
| 4, brain, outside scope | 120 | real brain MRI of a tumor the model has no class for |

Every subcategory is split into a **fit** half and an **eval** half. The
threshold is fitted on the fit half. Every rejection rate quoted in the report is
measured on the eval half, so it is not the number the threshold was tuned to.
Where images are slices out of a patient's scan, the split is by patient, so two
slices of the same head never straddle the boundary.

### Category 1: corruptions of images we already have

Free, no download, and they cover the most likely real-world failures: a bad
scan, a mis-set window, a wrong crop, a thumbnail pasted in by mistake.

Source images: 120 images sampled from the internal **train** split, stratified
by class, seeded. Never the test split. That matters. If the corruptions came
from test images, the false rejection rate measured on internal test would be
measuring a threshold that had already seen those images.

| subcategory | n | what was done |
|---|---|---|
| `blur_heavy` | 120 | Gaussian blur, radius 5.5% of the short side |
| `noise_heavy` | 120 | additive Gaussian noise, sigma 0.30 of full range |
| `near_black` | 120 | intensities scaled to 5% |
| `near_white` | 120 | intensities pushed to 95% of white |
| `underexposed` | 120 | gamma 3.5 |
| `overexposed` | 120 | gamma 0.22 |
| `corner_crop` | 120 | a 32% window from a random corner, upscaled; no brain, or the edge of one |
| `rotated_90` | 120 | rotated 90 or 270 degrees |
| `upside_down` | 120 | rotated 180 degrees |
| `jpeg_q5` | 120 | JPEG re-encoded at quality 5 |
| `thumbnail` | 120 | downsampled to 24x24 and blown back up |
| `solid_colour` | 120 | a single flat colour, generated |
| `pure_noise` | 120 | uniform random pixels, generated |

Licence: the corrupted images inherit whatever `data/brain_tumor` carries. The
two generated subcategories have no source.

Two notes on what is deliberately absent and deliberately present:

- **Horizontal flip is not here.** It is a training augmentation, so a
  left-right flipped scan is genuinely in scope. Including it would have
  manufactured an easy win.
- **`rotated_90` and `upside_down` still contain a readable brain.** They are
  the least clearly out-of-scope group in the whole set. Rejecting them is the
  right behaviour, because the model was never trained on them, but they are
  flagged separately in the report so they cannot pad the headline number.

### Category 2: non-brain medical images

Other body parts. The point is that a wrong-file upload in a clinic is far more
likely to be another scan than a photo of a dog.

| subcategory | n | source | licence |
|---|---|---|---|
| `abdominal_ct_axial` | 240 | MedMNIST v2 OrganAMNIST at 128px, axial abdominal CT from the Liver Tumor Segmentation Benchmark | CC BY 4.0 |
| `abdominal_ct_coronal` | 240 | MedMNIST v2 OrganCMNIST at 128px, coronal abdominal CT from the same source | CC BY 4.0 |
| `chest_xray` | 240 | MedMNIST v2 PneumoniaMNIST at 224px, paediatric chest X-ray | CC BY 4.0 |
| `breast_ultrasound` | 156 | MedMNIST v2 BreastMNIST at 224px | CC BY 4.0 |
| `pelvic_mri` | 84 | TCIA Prostate-3T, axial T2 pelvic MRI, 14 patients | CC BY 3.0 |
| `extremity_mri` | 72 | TCIA Soft-tissue-Sarcoma, axial MRI of limbs, 12 patients | CC BY 3.0 |

The two MRI subcategories are the ones that matter. They are the **same
modality** as the training data, so the model cannot fall back on "this does not
look like an MRI". The extremity scans in particular have the same geometry a
naive precheck keys on: one bright roughly round object, centred, on a black
field. They are the hardest honest test in this category.

The MedMNIST rows are all CT, X-ray and ultrasound, which are visibly different
from MRI and therefore easier.

**Gap:** no knee or spine **MRI** with a permissive licence was found in the
time available. The extremity MRI from Soft-tissue-Sarcoma includes thigh and
knee-level cross sections and is the closest substitute. Spine MRI is untested.

Citations:

- Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., Ni, B.
  (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification. *Scientific Data* 10, 41.
- Litjens, G., Futterer, J., Huisman, H. (2015). Data From Prostate-3T. The
  Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.QJTV5IL5
- Vallieres, M., Freeman, C. R., Skamene, S. R., El Naqa, I. (2015).
  Soft-tissue-Sarcoma. The Cancer Imaging Archive.
  https://doi.org/10.7937/K9/TCIA.2015.7GO2GSKS

### Category 3: non-medical images

The floor. Rejecting these proves close to nothing about clinical safety. They
are here because failing on them would mean something is badly broken.

| subcategory | n | source | licence |
|---|---|---|---|
| `photograph` | 240 | Imagenette v2 160px validation split, a 10-class subset of ImageNet | fast.ai wrapper Apache-2.0; underlying ImageNet photographs are research use only |
| `document` | 120 | generated: pages of printed text, as if a referral letter had been photographed | none, generated |
| `chart_screenshot` | 120 | generated: matplotlib line, bar, scatter and histogram figures | none, generated |

The ImageNet research-use restriction is the only licence in the whole set that
would block a commercial redistribution. It does not block this use, which is
evaluation. It is called out here so nobody bundles these images into a shipped
product.

### Category 4: brain MRI outside the trained scope

**This is the one that matters clinically, and it was obtainable.**

The model knows three tumor families plus no-tumor. It has no class for a
metastasis, a schwannoma, a stroke, an abscess or a haemorrhage. Handed any of
those it must return one of four wrong answers, and the dangerous one is
"no tumor".

| subcategory | n | source | licence |
|---|---|---|---|
| `vestibular_schwannoma` | 120 | TCIA Vestibular-Schwannoma-MC-RC, 20 patients, axial T1 contrast-enhanced and T2 | CC BY 4.0 |

Vestibular schwannoma is a tumor of the eighth cranial nerve. It is not a
glioma, not a meningioma and not a pituitary adenoma. The model has no class for
it. This collection is routine clinical imaging from ten UK sites, which is the
right kind of data: not a curated research set, the sort of scan a clinic
actually produces.

Citation: Kujawa, A., Dorent, R., Connor, S., Thomson, S., Ivory, M., Vahedi, A.,
Guilhem, E., Wijethilake, N., Bradford, R., Kitchen, N., Bisdas, S., Ourselin,
S., Vercauteren, T., Shapey, J. (2024). Vestibular-Schwannoma-MC-RC. The Cancer
Imaging Archive. https://doi.org/10.7937/HRZH-2N82

**Read this before quoting the category 4 number.** Three honest caveats:

1. **One tumor type, not four.** Metastases, stroke, abscess and haemorrhage are
   all still untested. Those datasets (BraTS-METS, ISLES, ATLAS) sit behind
   registration forms and could not be pulled non-interactively. Vestibular
   schwannoma is one point on a wide map.
2. **Twenty patients.** 120 images, but only 20 independent heads. The effective
   sample size for anything patient-level is 20, and the confidence intervals in
   the report are computed per image, so they are optimistic. Treat the category
   4 rate as an estimate with wide error bars.
3. **Slice level is a confound.** Vestibular schwannomas sit at the skull base,
   so these are posterior-fossa slices. The internal training data is mostly
   mid-brain. A rejection here could be the model refusing an unfamiliar tumor,
   or it could be the model refusing an unfamiliar slice level. The two cannot
   be separated with this data. The BRISC false rejection rate is the control
   that tells you how much of the effect is plain domain shift.

## Acquisition

Both steps write into `data/out_of_scope/_raw/`, which is gitignored.

### MedMNIST and Imagenette

```python
import medmnist
from medmnist import INFO
from torchvision.datasets import Imagenette

CACHE = "data/out_of_scope/_raw"
for flag, size in [("organamnist", 128), ("organcmnist", 128),
                   ("pneumoniamnist", 224), ("breastmnist", 224)]:
    cls = getattr(medmnist, INFO[flag]["python_class"])
    cls(split="test", download=True, size=size, root=CACHE)

Imagenette(root=CACHE + "/imagenette", split="val", size="160px", download=True)
```

About 1.7 GB. No account needed.

### TCIA

TCIA's NBIA REST API serves public collections without a login. For each
collection, list the MR series, take one series per patient, pull the zip, and
save six axial slices from the middle half of each volume as 8-bit PNG with a
1-99 percentile intensity window.

```
GET https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries
        ?Collection=<name>&Modality=MR
GET https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage
        ?SeriesInstanceUID=<uid>
```

Collections and series-description filters used:

| collection | filter | patients | slices each | destination |
|---|---|---|---|---|
| Vestibular-Schwannoma-MC-RC | description contains `t1` or `t2` | 24 requested, 20 kept | 6 | `_raw/tcia/cat4_vestibular_schwannoma` |
| Prostate-3T | `t2_tse_tra` | 14 | 6 | `_raw/tcia/cat2_pelvic_mri` |
| Soft-tissue-Sarcoma | `axial t1`, `axial fse`, `t1 -`, `stir` | 12 | 6 | `_raw/tcia/cat2_extremity_mri` |

Slices whose `ImageOrientationPatient` is not close to axial are dropped, which
is why fewer patients survive than were requested. About 700 MB.

The intensity windowing is worth being explicit about. Raw MRI DICOM values have
no fixed scale, so they must be windowed to be viewable at all. A per-slice
1-99 percentile stretch is used because it is the same kind of normalisation the
internal JPEGs already carry. Using raw values instead would have made these
images trivially rejectable for a reason that has nothing to do with anatomy.

## What is not in the set, and why that matters

- **Brain metastases.** The single most common brain tumor in adults, and the
  most likely thing a rural clinic will actually see that this model has no
  class for. Untested.
- **Stroke, haemorrhage, abscess.** Untested.
- **Post-operative brains.** Resection cavities look like nothing in the
  training set. Untested.
- **Paediatric brain MRI.** Untested.
- **Non-T1 sequences other than what BRISC and the schwannoma set carry.**
  FLAIR and diffusion-weighted imaging are untested.
- **Knee and spine MRI.** Extremity MRI is a partial substitute. Spine is
  untested.

Session E should carry this list into the model card unchanged.
