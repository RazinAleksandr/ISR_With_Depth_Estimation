from degradations import BSRDegradation


class AdvancedImageDegradation:
    def __init__(self):
        pass

    def apply_bsr_degradation(self, image):
        bsr_degradation = BSRDegradation()
        return bsr_degradation(image)
        