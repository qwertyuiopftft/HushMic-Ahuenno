# HushMic fine-tuned DPDFNet v4

This repository includes the optional binary
`assets/models/dpdfnet8_48khz_hushmic_finetuned_v4.onnx`.

## Important provenance notice

This checkpoint was fine-tuned from DPDFNet on a mixed public/private training
setup. The private training recordings and the complete dataset are **not**
included. The model is published by explicit permission of the speaker for
research and demonstration purposes.

Because private speech and room-noise material influenced the weights, this is
not a general-purpose or fully reproducible public benchmark model. It may
perform best on microphones, gain levels and rooms similar to the development
setup. Do not use it as a voice-identification system or as a substitute for
consent from people recorded in the room.

## Reported development result

On the private development validation run used for this fork (48 ten-second
clips), the candidate reported mean SI-SDR improvement of **11.01 dB**, mean
DNSMOS background score **4.09**, mean DNSMOS speech score **3.28**, and mean
DNSMOS overall score **3.02**. These numbers are directional only; the source
clips and evaluation scripts are not public.

## Usage

The model has the same streaming 48 kHz mono interface as DPDFNet-8. For a
manual experiment, point the denoiser at the file with `HUSHMIC_MODEL_PATH` or
copy it into a separate model directory. Keep the official DPDFNet model as a
fallback.

## Licensing

The architecture and pretrained starting point come from
[DPDFNet](https://github.com/ceva-ip/DPDFNet), licensed under Apache-2.0. The
fine-tuned checkpoint is provided under the same project licence where that
licence applies; the private training recordings are not redistributed.
