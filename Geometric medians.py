import scipy
import matplotlib.pyplot as plt
import numpy as np
import random
from random import sample
from itertools import product
from datetime import datetime

def geometric_mean(points):
    def sum_distance(q):
        value = 0
        for point in points:
            value += (((point[0] - q[0]) ** 2) + ((point[1] - q[1]) **2)) ** 0.5
        return value
    return scipy.optimize.minimize(sum_distance, [0,0], method="L-BFGS-B")


def plot_random_points(n):
    x = sample(range(0,100), n)
    y = sample(range(0,100), n)

    points = [i for i in zip(x, y)]

    opt_x = geometric_mean(points).x[0]
    opt_y = geometric_mean(points).x[1]

    for i in range(len(x)):
        plt.plot([opt_x, x[i]],[opt_y,y[i]], 'y:')

    plt.plot(x, y, '.r')
    plt.plot(opt_x, opt_y, 'ob')


'Inspiration hentet fra: https://stackoverflow.com/questions/45766534/finding-cross-product-to-find-points-above-below-a-line-in-matplotlib'
def cross_product(a, b, c):
    return np.cross(np.array(b)-np.array(a), np.array(c)-np.array(a)) # Vi bruger numpy's funktion cross til at regne krydsproduktet


def two_geometric_medians(points):
    def sum_distance_(q, points): 
        Euclidean_distance = 0
        for point in points:
            Euclidean_distance += (((point[0] - q[0]) ** 2) + ((point[1] - q[1]) **2)) ** 0.5
        return Euclidean_distance

    n = len(points)
    
    total_dist = []
    lowest_distance = np.inf
    
    #Vi laver to lister der vil komme til at indeholde punkterne omkring q_1 (til venstre for bisektoren) og omkring q_2 (til højre for bisektoren)
    points_left = []
    points_right = []
    
    for i,j in product(range(n), range(n)): # Vi laver et nested for-loop med itertools 'product'
        L = []
        R = []
        
        for p in range(n):
            
            if cross_product(points[i],points[j],points[p]) < 0: # Negativt krydsprodukt --> punktet ligger til venstre
                L.append(points[p])
        
            elif cross_product(points[i],points[j],points[p]) >= 0:  # Positivt krydsprodukt --> punktet ligger til højre
                R.append(points[p])
    
        #Vi regner nu de geometriske gennemsnit for de punkter vi har fundet til venstre og til højre.
        q_1 = geometric_mean(L).x
        q_2 = geometric_mean(R).x
        
        #Vi kalder nu funktionen sum_distance_(q, points) for punkterne til højre og puntkerne til venstre:
        total_dist.append(sum_distance_(q_1 , L ) + sum_distance_(q_2 , R))
        
        # vi vil nu gerne finde de to punkter der minimierer summen af afstanden, så vi bruger nu np.inf til at finde den mindste: 
        'Inspiration hentet fra: https://stackoverflow.com/questions/34264710/what-is-the-point-of-floatinf-in-python'
        for dist in total_dist:
            if dist < lowest_distance:
                lowest_distance = dist
                if lowest_distance == min(total_dist): #Når vi har fundet den laveste sum har vi også den endelige inddeling af højre og venstre:
                    points_left  = L
                    points_right = R

    # Vi finder nu de geometriske gennemsnit for højre og venstre siden rundet af til 3 decimaler
    q_1x, q_1y = np.round(geometric_mean(points_left).x, 3)
    q_2x, q_2y = np.round(geometric_mean(points_right).x, 3)

    'Vi laver nu en dictionary: inspiration hentet fra: https://www.geeksforgeeks.org/g-fact-41-multiple-return-values-in-python/'
    
    def return_variables():
        d = dict();
        d['Left'] = points_left
        d['Right'] = points_right
        d['Medians'] = [(q_1x, q_1y), (q_2x, q_2y)]
        return d
    d = return_variables()

    return d


def bisector(p_1,p_2):
    mid_point = (p_1[0] + p_2[0]) /2 , (p_1[1] + p_2[1]) /2
    
    slope_of_segment = (p_1[1] - p_2[1]) / (p_1[0] - p_2[0])
    slope_bisector = - np.reciprocal(slope_of_segment) # negative reciprokke værdi af vores linje-segments hældning
    
    intercept = mid_point[1] - (slope_bisector * mid_point[0])
    
    bisecting_line = [slope_bisector, intercept, mid_point]
    return bisecting_line


def plot_medians(n):
    # Vi generer en række tilfældige punkter, som skalereres fra 0-1.
    points = [(round(random.random(),5), round(random.random(),5)) for _ in range(n)]

    x = np.linspace(-1 , 1) 
    y = x

    dictio = two_geometric_medians(points)

    Left = dictio.get('Left')
    two_medians = dictio.get('Medians')

    q_1 = np.array(two_medians[0])
    q_2 = np.array(two_medians[1])

    L = bisector(q_1,q_2)
    m = np.round(L[0],3)
    b = np.round(L[1],3)
    mid_point = np.round(L[2],3)

    #vi tilføjer bisektoren til plottet
    plt.plot(y,m*x+b, 'blue')

    #vi tegner linjerne fra punkter til hhv q_1 og q_2
    for p in points:
        if p in Left:
            plt.plot([p[0],q_1[0]],[p[1],q_1[1]],'y:')
        else:
            plt.plot([p[0],q_2[0]],[p[1],q_2[1]],'y:')   

    #vi plotter de to punkter q_1 og q_2, midtpunktet og de andre punkter
    plt.plot([i[0] for i in two_medians], [i[1] for i in two_medians], marker ='o', color = 'blue', linestyle="--")
    plt.scatter([i[0] for i in points], [i[1] for i in points],marker = '.', color='red')
    plt.plot(mid_point[0], mid_point[1], marker ='o', color = 'blue')
     
    #plt.title("Two Geometric Medians", size=15, color = 'black', fontweight ="bold")
    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 1])


def plot_specific_points(points):
    x = []
    y = []

    for point in points:
        x.append(point[0])
        y.append(point[1])

    opt_x = geometric_mean(points).x[0]
    opt_y = geometric_mean(points).x[1]

    for i in range(len(x)):
        plt.plot([opt_x, x[i]],[opt_y,y[i]], 'y:')

    plt.plot(x, y, '.r')
    plt.plot(opt_x, opt_y, 'ob')


def plot_with_contour(points):
    def sum_distance(q):
        value = 0
        for point in points:
            value += (((point[0] - q[0]) ** 2) + ((point[1] - q[1]) **2)) ** 0.5
        return value
    
    opt_x = geometric_mean(points).x[0]
    opt_y = geometric_mean(points).x[1]

    x_values = [x for x, y in points]
    y_values = [y for x, y in points]

    min_x = min(x_values)
    min_y = min(y_values)
    max_x = max(x_values)
    max_y = max(y_values)

    z_values = [[sum_distance((x, y)) for x in np.linspace(min_x-1, max_x+1, 50)] for y in np.linspace(min_y-1, max_y+1, 50)]
    
    qcs = plt.contour(z_values, levels = 10, extent = [min_x-1, max_x+1, min_y-1, max_y+1],cmap = "plasma")
    for i in range(len(x_values)):
        plt.plot([opt_x, x_values[i]],[opt_y, y_values[i]], 'y:')
        
    plt.plot(x_values,y_values, ".r")
    plt.plot(opt_x,opt_y,"ob")

    plt.clabel(qcs, fontsize=8, fmt="%.1f")


def add_title(title):
    plt.title(title, size=10, color = 'black')


def run_script():
    start_time = datetime.now()

    plt.subplot(2,3,1)
    plot_random_points(10)
    add_title("10 Random Points")

    plt.subplot(2,3,2)
    plot_specific_points([[1.0,1.0],[3.0,3.0]])
    add_title("2 Hardcoded Points")

    plt.subplot(2,3,3)
    plot_specific_points([[2,9],[92,32],[54,87]])
    add_title("3 Hardcoded Points")

    plt.subplot(2,3,4)
    plot_with_contour([[1,2],[5,0.5],[6,5]])
    add_title("3 Hardcoded Points with Contour")

    plt.subplot(2,3,5)
    plot_with_contour([[1,1],[1,3],[1,8],[2,3],[7,4],[2,5],[9,6],[2,7],[3,9],[9,9]])
    add_title("10 Hardcoded Points with Contour")

    plt.subplot(2,3,6)
    plot_medians(10)
    add_title("10 Random Points with Geometric Medians")

    plt.subplots_adjust(wspace=0.4,hspace=0.4)

    end_time = datetime.now()
    print('Time to run script: {}'.format(end_time - start_time))    

    plt.show()


run_script()
