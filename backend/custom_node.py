
from backend.workflow_procedure import workflow


class video_translation:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_url": ("STRING", {"multiline": False}),
                "target_language": ("STRING", {"default": "en"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    
    FUNCTION = "process"
    
    OUTPUT_NODE = True
    
    CATEGORY = "Translation"
    
    EXCUTE = "process"
    
    @staticmethod
    def process(video_url, target_language) -> str:
        video_final_name = workflow(video_url, target_language)
        return (video_final_name)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "videotranslation": video_translation,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "videotranslation": "Video Translation",
}