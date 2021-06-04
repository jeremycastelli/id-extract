class Keypoint_obj():
    def __init__(self, x,y, width=80, height=80, color='#38a8a4'):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.selected = False
        
    def set_x_y(self,x_in,y_in)  :
        self.x = x_in
        self.y = y_in
    
    def draw(self, canvas = None):
        canvas.fill_style = self.color
        canvas.fill_rect(self.x - (5), self.y - (5) , 10, 10)
       
    def is_selected(self,x_in, y_in):
        x_coord = self.x - (self.width*0.5)
        y_coord = self.y - (self.height*0.5)

        if x_in > x_coord and x_in < (x_coord+ self.width) and  y_in > y_coord  and y_in < (y_coord  + self.height):
            self.set_selected(True)
            return True
        else:
            self.set_selected(False)
            return False
    
    def set_selected(self,state):
        self.selected = state