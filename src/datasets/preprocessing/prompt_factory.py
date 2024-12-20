import torch


def tab2prompt_breast_lesion(model_name, phase, batch_data, txt_processor,
                             llava_mm_use_im_start_end=None, llava_conv_mode=None, answer_mode='short', _context_length=None, _group_age=None):
    text_samples = []

    if batch_data["dataset_name"][0] == "cbis-ddsm":
        for breast_side, breast_density, abnormal, mass_shape, mass_margin, calc_morph, calc_dist in \
            zip(batch_data["breast_side"], batch_data["breast_density"], \
                batch_data["abnormal"], batch_data["mass_shape"], batch_data["mass_margin"], \
                batch_data["calc_morph"], batch_data["calc_dist"]):                                                        

            if abnormal == 'mass':
                if mass_margin.lower() == '-1':
                    text =  f"A mass lesion with {mass_shape.lower()} shape " \
                        f"is located in the {breast_side.lower()} breast. This {breast_side.lower()} breast has {breast_density.lower()} density."
                else:                    
                    text =  f"A mass lesion with {mass_shape.lower()} shape and {mass_margin.lower()} margin " \
                            f"is located in the {breast_side.lower()} breast. This {breast_side.lower()} breast has {breast_density.lower()} density."
            elif abnormal == 'calcification':
                text =  f"A calcification lesion with {calc_morph.lower()} appearance and {calc_dist.lower()} distribution " \
                        f"is located in the {breast_side.lower()} breast. This {breast_side.lower()} breast has {breast_density.lower()} density."
                
            text_samples.append(text)
    elif batch_data["dataset_name"][0] == "embed":
        for tissueden, breast_side, mass_shape, mass_margin, mass_dense, calc_morph, calc_dist, marital, ethnicity, ethnic_group, age, findings in \
            zip(batch_data["tissueden"], batch_data["breast_side"], \
                batch_data["mass_shape"], batch_data["mass_margin"], batch_data["mass_dense"], \
                batch_data["calc_morph"], batch_data["calc_dist"], \
                batch_data["marital_status"], batch_data["ethnicity"], batch_data["ethnic_group"], \
                batch_data["age"], batch_data["findings"]):
            
            # Patient demographic information
            text = f"A patient"
            if marital != "-1":
                text += f", whose marital status is {marital.lower()}"
            if ethnicity != "-1":
                text += f", whose ethnicity is {ethnicity.lower()}"
            if ethnic_group != "-1":
                text += f", whose ethnic group is {ethnic_group.lower()}"
            if age != "-1":
                round_age = int(round(age.item()))
                if _group_age is None:
                    text += f", whose age is {round_age}"
                elif _group_age == '3_groups':
                    # grouped based on this: https://www.cancer.org/cancer/types/breast-cancer/about/how-common-is-breast-cancer.html
                    if round_age < 45:
                        age_group = 'young adults'
                    elif round_age < 63:
                        age_group = 'middle-aged adults'
                    else:
                        age_group = 'old adults'
                    text += f", who is in {age_group} age group"
                else:
                    raise ValueError(f"[!] _group_age argument must receive values from [None, '3_groups'], receiving {_group_age} instead...")
            text += '. '

            # Findings information
            # Breast side
            if findings != "-1":
                text += f"There are {findings.lower()} findings in {breast_side.lower()}. "
            else:
                text += f"There are findings in {breast_side.lower()}. "

            # Breast density
            if tissueden != "-1":
                text += f"{breast_side.capitalize()} density is {tissueden.lower()}. "
            
            # Mass findings
            if mass_shape != "-1" or mass_margin != "-1" or mass_dense != "-1":
                text += f"There are masses"

                if mass_shape != "-1":
                    text += f", with {mass_shape.lower()} shape"
                if mass_margin != "-1":
                    text += f", with {mass_margin.lower()} margin"
                if mass_dense != "-1":
                    text += f", with {mass_dense.lower()}" # not typo, no need for the word "dense"                
                text += '. '
            
            # Calcification findings
            if calc_morph != "-1" or calc_dist != "-1":
                text += f"There are calcifications"

                if calc_morph != "-1":
                    text += f", with {calc_morph.lower()} appearance"
                if calc_dist != "-1":
                    text += f", with {calc_dist.lower()} distribution"
                text += '. '

            # print(text)
            # print()
            text_samples.append(text)

    
    processed_text_samples = None
    if model_name in ['lavis_blip2']:
        _txt_processor = txt_processor['train'] if phase == 'train' else txt_processor['eval']
        processed_text_samples = [_txt_processor(text_sample) for text_sample in text_samples]
    elif model_name in ['open_clip']:
        _txt_processor = txt_processor

        if _context_length is not None:
            processed_text_samples = _txt_processor(text_samples, context_length=_context_length)
        else:
            processed_text_samples = _txt_processor(text_samples)
    elif model_name in ["pubmed_clip"]:
        _txt_processor = txt_processor

        processed_text_samples = _txt_processor(text_samples, return_tensors="pt", padding=True)
        processed_text_samples = processed_text_samples['input_ids']

    elif model_name in ['llava']:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates

        from llava.mm_utils import tokenizer_image_token

        processed_text_samples = []
        for qs in text_samples:
            
            if answer_mode == 'long':
                qs = qs + ' ' + "Does this patient have high risk of breast cancer?"
            elif answer_mode == 'short':
                qs = qs + ' ' + "Does this patient have high risk of breast cancer?"
                qs = qs + "\nAnswer the question using a single word or phrase."
            elif answer_mode == 'multiple-choice':
                qs = qs + ' ' + "Classify this patient into either benign or malignant?"
                qs = qs + "\nA. Benign\nB. Malignant\nAnswer with the option's letter from the given choices directly."
            elif answer_mode == 'multiple-choice-v2':
                qs = qs + ' ' + "Does this patient have high risk of breast cancer? Classify this patient into either benign or malignant."
                qs = qs + "\nA. Benign\nB. Malignant\nAnswer with the option's letter from the given choices directly."
            elif answer_mode == 'multiple-choice-v3':
                qs = qs + ' ' + "What is the expected pathology outcome if the patient undergoes a biopsy?"
                qs = qs + "\nA. Benign\nB. Malignant\nAnswer with the option's letter from the given choices directly."
            elif answer_mode == 'multiple-choice-v4':
                qs = qs + ' ' + "What potential results might we obtain from conducting a biopsy on the patient?"
                qs = qs + "\nA. Benign\nB. Malignant\nAnswer with the option's letter from the given choices directly."
            elif answer_mode == 'multiple-choice-v5':
                qs = qs + ' ' + "Considering the mammogram findings and related radiologic features, what might be the expected pathology outcome if the patient undergoes a biopsy?"
                qs = qs + "\nA. Benign\nB. Malignant\nAnswer with the option's letter from the given choices directly."
            else:
                raise ValueError("Variable `answer_mode` receives unrecognized value!")

            if llava_mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[llava_conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, txt_processor, IMAGE_TOKEN_INDEX, return_tensors='pt')            

            processed_text_samples.append(input_ids)
    else:
        raise ValueError(f"Have not handled model {model_name} yet!")

    
    return processed_text_samples, text_samples