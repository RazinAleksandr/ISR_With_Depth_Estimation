from degradations import BSRDegradation, ImresizeDownscale


class AdvancedImageDegradation:
    def __init__(self):
        pass

    def apply_bsr_degradation(self, image):
        bsr_degradation = BSRDegradation()
        return bsr_degradation(image)
    
    def apply_resize_degradation(self, image):
        resize_degradation = ImresizeDownscale()
        return resize_degradation(image)
        