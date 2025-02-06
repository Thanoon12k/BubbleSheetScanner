class FiltersManager:
    def __init__(self, bubbles):
        self.bubbles = bubbles
        self.filtered_bubbles=None

        def filter_by_radious(self,thresh=3):
            bubbles=self.filtered_bubbles
            if self.filtered_bubbles==None:
                bubbles=self.bubbles
            radiuses = [cv2.boundingRect(bubble)[2] for bubble in bubbles]
            most_common_radius = max(set(radiuses), key=radiuses.count)
            print(f'Most common radius: {most_common_radius} with {radiuses.count(most_common_radius)} occurrences')
            
            filterd_bubbles = [bubble for bubble in bubbles if abs(cv2.boundingRect(bubble)[2] - most_common_radius) <= thresh]
            self.filtered_bubbles=filterd_bubbles
        return filterd_bubbles

        
            
            
