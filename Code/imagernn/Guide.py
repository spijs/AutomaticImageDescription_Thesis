__author__ = 'Wout&Thijs'


class guide:

    def get_guide_size(guide_type):
        if guide_type=="image":
            return 256

    def get_guide(self,guide_type,im):
        if guide_type=="image":
            return self.get_image_guide(im)

    def get_image_guide(self,im):
        return im

