from haystack import component

@component
class QuestionCutter:
	@component.output_types(out_text=list[str])
	def run(self, in_text:list[str]):
		return {"out_text": [t.split("Question:")[0] for t in in_text]}