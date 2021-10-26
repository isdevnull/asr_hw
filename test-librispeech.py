import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.datasets.utils import get_dataloaders
from hw_asr.metric.utils import calc_wer, calc_cer
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # text_encoder
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = []
    cer_list_greedy = []
    cer_list_beam = []
    wer_list_beam = []
    wer_list_greedy = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["val"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[:int(batch["log_probs_length"][i])]
                results.append(
                    {
                        "ground_trurh": batch["text"][i],
                        "pred_text_argmax": text_encoder.ctc_decode(argmax),
                        "pred_text_beam_search": text_encoder.ctc_beam_search(
                            batch["probs"][i], batch["log_probs_length"][i], beam_size=10
                        )[:10],
                    }
                )
                target = results[-1]["ground_truth"]
                pred_greedy = results[-1]["pred_text_argmax"]
                pred_beam = results[-1]["pred_text_beam_search"][0]

                cur_wer_beam = calc_wer(target, pred_beam)
                cur_cer_beam = calc_cer(target, pred_beam)
                cur_cer_greedy = calc_cer(target, pred_greedy)
                cur_wer_greedy = calc_wer(target, pred_greedy)
                print(f"Beam size = {100}; WER: {cur_wer_beam}; CER: {cur_cer_beam}")
                print(f"Greedy; WER: {cur_wer_greedy}; CER: {cur_cer_greedy}")
                wer_list_beam.append(cur_wer_beam)
                wer_list_greedy.append(cur_wer_greedy)
                cer_list_beam.append(cur_cer_beam)
                cer_list_greedy.append(cur_wer_greedy)
    print(f"Greedy WER: {sum(wer_list_greedy) / len(wer_list_greedy)}")
    print(f"Beam WER: {sum(wer_list_beam) / len(wer_list_beam)}")
    print(f"Beam CER: {sum(cer_list_beam) / len(cer_list_beam)}")
    print(f"Greedy CER: {sum(cer_list_greedy) / len(cer_list_greedy)}")
    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        #test_data_folder = Path(args.test_data_folder).absolute().resolve()
        #assert test_data_folder.exists()
        config.config["data"] = {
            "val": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "LibrispeechDataset",
                        "args": {
                            "part": args.test_data_folder
                        }
                    },
                ],
            }
        }

    assert config.config.get("data", {}).get("val", None) is not None
    config["data"]["val"]["batch_size"] = args.batch_size

    main(config, args.output)