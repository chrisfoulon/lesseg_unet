#!/bin/bash
# Test multi-modal training (DWI + ADC) with CPU mode

python -m lesseg_unet.main \
  -d cpu \
  -p /home/chrisfoulon/neuro_data/sub_sample/ISLES_BBS/dwi /home/chrisfoulon/neuro_data/sub_sample/ISLES_BBS/adc \
  -imn dwi adc \
  -lp /home/chrisfoulon/neuro_data/sub_sample/ISLES_BBS/stroke \
  -lmn stroke \
  -o /home/chrisfoulon/neuro_apps/lesseg_unet/output_test_multimodal \
  -nf 3 \
  --subject-pattern '(sub-\d+)'
