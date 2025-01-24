"""
Custom haystack component to remove "Question:" and everything after.
No need to run this.
"""

from haystack import component

@component
class QuestionCutter:
	@component.output_types(out_text=list[str])
	def run(self, in_text:list[str]):
		# Split each input by "Question:" and return the 0th element. So return everything before "Question:"
		return {"out_text": [t.split("Question:")[0] for t in in_text]}