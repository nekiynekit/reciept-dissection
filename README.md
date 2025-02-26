# reciept-dissection
Repo with OCR reciept dissection node for tg-bot

# Ressearch
Looks like there are no any type of russian receipt dataset for simple OCR training. So there're two nice datasets i found - CORD and SPOIE. CORD has nice convex hull around words in receipt, row_id, category, data and so on. Using this bounding areas i want to train some simple to predict something like [`x1`, `y1`, ..., `x4`, `y4`] + [`row_idx_from_top`, `category` (price / name)]. 

Not sure if `row_idx_from_top` is nice target to predict, but has no idea how to do it in better way.

On inference, after we predict areas - just cut these bounding box from image and sent to any russian-language model to predict values and names.

???
PROFIT
