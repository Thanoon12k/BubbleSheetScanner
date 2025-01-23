from onefile import *
import cv2

def getBubblesContours(img, adaptiveFrame, expected_count=540,min_ratio=0,max_ratio=0):
    contours, hierarchy = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # min_ratio = 0.99
    # max_ratio = 1.01
    circular_contours = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) > 2:  # Assuming circular shapes have more than 3 vertices
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(h) / w
            condition1= min_ratio <= aspect_ratio <= max_ratio and cv2.contourArea(contour) > 30  #aspect ration = h/w
            condition2=not ((x >= img.shape[1] // 4 and y <= img.shape[0] // 2.9) or (y <= img.shape[0] // 7)) # not in top right quarter
                               
            if condition1 and condition2:  # Remove small dots
                circular_contours.append(contour)

    contours = circular_contours
    if contours:
        average_area = sum(cv2.contourArea(contour) for contour in contours) / len(contours)
    else:
        average_area = 0
    filtered_contours = []
    for cnt in contours:
        if average_area / 2 <= cv2.contourArea(cnt) <= average_area * 2:
            filtered_contours.append(cnt)
    contours = filtered_contours 
    return contours
def have_two_twins(fm,cntr1,cntr2):
        x1, y1, w,_ = cv2.boundingRect(cntr1)
        x2, y2, _,_ = cv2.boundingRect(cntr2)
        
        max_normlized_value=w
        twiny=False
        if abs(x1-x2)<max_normlized_value: twiny=True
        if abs(y1-y2)<max_normlized_value:twiny=True
        print(f"twiny: {twiny}")
        if twiny==False:
            print('false twiny')
            display_images([draw_contours_on_frame(fm,[cntr1,cntr2],color='b',add_colors=True)],scale=50)
        return twiny





def remove_not_aligned_counters(fm,counters):
    aligned_contours=[]
    counters = sorted(counters, key=lambda c: cv2.boundingRect(c)[1])

    for i in range(0,len(counters)-1,2):
        if is_twin(fm,counters[i],counters[i+1]):
            aligned_contours.append(counters[i])
            aligned_contours.append(counters[i+1])
            
    return aligned_contours


def get_number_bubbles(bw_img,cnts:list)->list:
    number_cnts=[]
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
    cnts=cnts[:40]
    # Choose the first 4 contours and sort them by x-coordinate
    bubble_columns=4
    bubbles_rows=10
    first_four_cnts = sorted(cnts[:bubble_columns], key=lambda c: cv2.boundingRect(c)[0])
    
    x_first, _, _,_ = cv2.boundingRect(first_four_cnts[0])
    x_last, _, w, _ = cv2.boundingRect(first_four_cnts[-1])

    normlize_value=w//2
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    first_ten_cnts=sorted(cnts[:bubbles_rows], key=lambda c: cv2.boundingRect(c)[1])
    
    _, y_first, _, _ = cv2.boundingRect(first_ten_cnts[0])
    _, y_last, _, _ = cv2.boundingRect(first_ten_cnts[-1])

    

    # display_images([draw_contours_on_frame(bw_img,[first_ten_cnts[0]],add_colors=True)],scale=50)
    # display_images([draw_contours_on_frame(bw_img,[first_ten_cnts[-1]],add_colors=True)],scale=50)
     # remove our of y
    for cntr in cnts:
        _,y,_,_=cv2.boundingRect(cntr)
        if y>= (y_first-normlize_value)  and y<=(y_last+normlize_value):
            number_cnts.append(cntr)
   
    # remove our of x
    # for cntr in cnts:
    #     x,_,_,_=cv2.boundingRect(cntr)
    #     if x>= (x_first-normlize_value)  and x<=(x_last+normlize_value):
    #         number_cnts.append(cntr)
    # print(f'xfirst: {x_first}      xlast: {x_last}        yfirst: {y_first}  ylast: {y_last}  num_contours: {len(number_cnts)}')
    # Print the coordinates on the left top corner of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'x_first: {x_first}, x_last: {x_last}, y_first: {y_first}, y_last: {y_last}, num_contours: {len(number_cnts)}'
    cv2.putText(bw_img, text, (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    draw_contours_on_frame(bw_img,number_cnts,display=True,add_colors=True)
    
    return number_cnts



def add_miss_bubbles_to_row(fm,ref,row,normlize_value):
    ref = sorted(ref, key=lambda c: cv2.boundingRect(c)[0])
    row = sorted(row, key=lambda c: cv2.boundingRect(c)[0])
    filtered_row = []
    added_buubles=[]
    first_bubble_y = sorted(row, key=lambda c: cv2.boundingRect(c)[1])
    first_bubble_y = cv2.boundingRect(first_bubble_y[0])[1]
    for c in row:
        y = cv2.boundingRect(c)[1]
        if abs(y - first_bubble_y) < normlize_value:
            filtered_row.append(c)
    row = filtered_row
    # draw_contours_on_frame(fm,ref,color='g',display=True,add_colors=True)
    # draw_contours_on_frame(fm,row,color='b',display=True,add_colors=True)
    while len(row)<4:
        for i in range(0,3):
            equal_X=abs(cv2.boundingRect(ref[i])[0]-cv2.boundingRect(row[i])[0]) <normlize_value
            if equal_X:
                print('equal: ',i)
            else:
                print('not equal: ',i)
                new_bubble = ref[i].copy()
                for point in new_bubble:
                    point[0][1] += int(normlize_value * 2.5)        
                row.insert(i, new_bubble)
                added_buubles.append(new_bubble)
                draw_contours_on_frame(fm,row,color='g',display=True,add_colors=True)
                break

    

    return added_buubles

def fix_missing_num_bubbles(bw_img, cnts: list) -> list:
    Refererece_row = []
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
    _, _, w, _ = cv2.boundingRect(cnts[0])
    normlize_value = w // 2

    while len(cnts) < 40:
        for i in range(0, len(cnts) - 4, 4):
            contours_row = sorted(cnts[i:i + 4], key=lambda c: cv2.boundingRect(c)[1])
            y_values = [cv2.boundingRect(c)[1] for c in contours_row]
            is_not_good_row = max(y_values) - min(y_values) > normlize_value

            draw_contours_on_frame(bw_img, [cnts[i], cnts[i + 1], cnts[i + 2], cnts[i + 3]], display=True, add_colors=True)
            
            if not is_not_good_row and Refererece_row == []:Refererece_row = contours_row
            
            if is_not_good_row:
                miss_bubbles = add_miss_bubbles_to_row(bw_img, Refererece_row, contours_row, normlize_value)
                cnts.extend(miss_bubbles)
                i = 0
                break

        print(f"i->{i} from 40 new_rows= {len(cnts)}")
    cnts=sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])    
    return cnts

# Make sure to call the function with actual data, e.g.,
# fixed_rows = fix_missing_num_bubbles(bw_img, cnts)



def remove_not_aligned_bubbls(cnts: list) -> list:
    filterd_cnts=[]
    return cnts

def draw_rect_top_right_quarter(page):
    h, w, _ = page.shape
   
    cv2.rectangle(page,  (w // 4, 0)   , (w, int(h // 2.9)), 255, 14)  
    cv2.rectangle(page, (0,0), (w,h//7), 255, 14)  
    return page
def ggetAdaptiveThresh(frame,maxx=99,minn=9):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, maxx, minn)
