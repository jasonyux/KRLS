import json
import argparse

from mwzeval.metrics import Evaluator

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-input", type=str, required=True)
	args = parser.parse_args()
	with open(args.input, "r") as f:
		my_predictions = json.load(f)
	if "predictions" in my_predictions:  # some logged intermediate results has a key "predictions"
		my_predictions = my_predictions["predictions"]

	e = Evaluator(bleu=True, success=True, richness=True, dst=True)
		
	score = e.evaluate(my_predictions)
	print(json.dumps(score, indent=4))
	total_score = score['bleu']['mwz22'] + (score['success']['inform']['total'] + score['success']['success']['total']) / 2.0
	print(f"Total score: {total_score:.3f}")