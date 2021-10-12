# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1.0
    lev_dist = editdistance.eval(predicted_text, target_text)
    return lev_dist / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_parts, predicted_parts = target_text.split(), predicted_text.split()
    if len(target_parts) == 0:
        return 1.0
    lev_dist = editdistance.eval(predicted_parts, target_parts)
    return lev_dist / len(target_parts)
