import argparse
import os
import json
from baselines.utils.preprocessing import preprocess_and_save
from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
from baselines.sample_generate.generate import generate_data
from baselines.all_baselines import run_all_baselines



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) 
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="5")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-small")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--DEVICE', type=str, default ='cuda')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--prompt_len', type=int, default=30)
    parser.add_argument('--generation_len', type=int, default=200)
    parser.add_argument('--min_words', type=int, default=55) 
    parser.add_argument('--min_len', type=int, default=150) 
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--baselines', type=str, default="DetectGPT,NPR")
    args = parser.parse_args()
    

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples
    cache_dir, base_model_name, SAVE_FOLDER = preprocess_and_save(args)
    model_config ={}
    model_config['cache_dir'] = cache_dir
    # generic generative model
    model_config = load_base_model_and_tokenizer(args, model_config)

    # mask filling t5 model
    model_config = load_mask_filling_model(args, mask_filling_model_name, model_config)
    print(f'Loading dataset {args.dataset}...')
    data = generate_data(args, model_config)
    # write the data to a json file in the save folder
    baselines = [x for x in args.baselines.split(',')]
    baseline_outputs = run_all_baselines(data, args, n_perturbation_list, model_config, baselines= baselines)
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    with open(os.path.join(SAVE_FOLDER, "result.json"), "w") as f:
        json.dump(baseline_outputs, f, indent=4)

    
        

    

    

