from haystack import component

@component
class QuestionCutter:
	@component.output_types(out_text=str)
	def run(self, in_text:str):
		return {"out_text": in_text.split("Question:")[0]}