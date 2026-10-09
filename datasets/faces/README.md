# Face Dataset

## Purpose

This directory holds face images for **enrollment** — building the known-person
embedding database used by the face recognition pipeline.

**No real identities are included here.** Add your own using generic names.

---

## Directory Structure

```
datasets/faces/known/
    person_001/
        image1.jpg
        image2.jpg
        image3.jpg
    person_002/
        photo1.jpg
        photo2.png
    ...
```

Each **subdirectory name** is the identity label that will be spoken aloud
and stored in the embedding database. Use any identifier you choose.

---

## How to Add a Known Person

1. Create a subdirectory with a meaningful name:
   ```bash
   mkdir -p datasets/faces/known/alice
   ```
2. Place 3–10 face photos in the directory:
   ```bash
   cp /path/to/alice_photos/*.jpg datasets/faces/known/alice/
   ```
3. Re-run enrollment:
   ```bash
   python scripts/enroll_faces.py
   ```

---

## Image Guidelines

| Factor | Recommendation |
|--------|---------------|
| Format | JPEG or PNG |
| Resolution | At least 160×160 pixels (face crop) |
| Lighting | Front-lit, minimal shadows |
| Angle | Frontal and slight angle variations help robustness |
| Quantity | 5–10 images per person is a good baseline |
| Glasses | Include some images with glasses if the person wears them |
| Background | Plain background preferred but not required |

---

## Privacy

- This directory is in `.gitignore` — do NOT commit face images to a repository.
- Face images of real people should be handled according to your local privacy laws.
- The `.pkl` embedding database (`data/face_embeddings/embeddings.pkl`) contains
  mathematical representations of faces and should also be treated as personal data.

---

## Sample Placeholder

Two placeholder directories (`person_001`, `person_002`) are provided to show
the expected structure. They are empty — add your own images.

---

## What happens if no faces are enrolled?

The system will still run. Any detected face will be classified as UNKNOWN,
and the speech alert will say "Unknown person nearby".
