import json
import src.Commons.Models.SpectralAnalysisModel as sam


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, sam.SpectralAnalysisModel):
            return {"file_name": obj.file_name,
                    "ID": obj.ID,
                    "agent": obj.agent,
                    "concentration": obj.concentration,
                    "iteration": obj.iteration,
                    "data_x": obj.data_x.tolist(),
                    "data_y": obj.data_y.tolist()}
        # Let the base class handle the problem.
        return json.JSONEncoder.default(self, obj)
