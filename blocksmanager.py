import cv2

class BlocksManager:
    def __init__(self):
        self.bubbles=[]
        self.blocks_bubbles_dict = {}
        self.num_blocks = 0
        self.blocks_widths_dict = {}
        self.blocks_heights_dict = {}
        self.average_x_space=0
        self.average_y_space=0

        def get_blocks(self):
            av_x_space, av_y_space = self.average_x_space, self.average_y_space
            normlized_value=av_x_space*2
            cnts =self.bubbles
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if w >= normlized_value and h >= normlized_value:
                    self.num_blocks += 1
                    self.blocks_bubbles_dict[self.num_blocks]=[]
                    self.blocks_widths_dict[self.num_blocks]=c
                else:
                    self.blocks_bubbles_dict[self.num_blocks].append(c)
            
            return self.blocks_bubbles_dict

            def get_blocks_widths_height(self):
                blocks=self.blocks_bubbles_dict
                width=0
                i=0
                for block in blocks:
                    i+=1
                    cnts=sorted(blocks[block], key=lambda x: cv2.boundingRect(x)[0])
                    min_x = cv2.boundingRect(cnts[0])[0]
                    max_x = cv2.boundingRect(cnts[-1])[0]
                    cnts=sorted(blocks[block], key=lambda x: cv2.boundingRect(x)[1])
                    min_y = cv2.boundingRect(cnts[0])[1]
                    max_y = cv2.boundingRect(cnts[-1])[1]
                    height = max_y-min_y
                    width = max_x-min_x

                    self.blocks_heights_dict[i]=height
                    self.blocks_widths_dict[i]=width
                return self.blocks_widths_dict, self.blocks_heights_dict