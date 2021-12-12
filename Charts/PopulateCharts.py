import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


class charts:

    def __init__(self) -> None:
        pass

    def create_charts(self, mnist, verified, scrape):

        N = 2
        ind = np.arange(N) 
        width = 0.25

        mnistModel = mnist
        verifiedModel = verified
        scrapedModel = scrape

        plt.title("Acurracy for Models")
        

        X = ['Training','Test']
        
        X_axis = np.arange(len(X))

        max_y_lim = 1
        min_y_lim = 0
        plt.ylim(min_y_lim, max_y_lim)

        bar1 = plt.bar(ind, mnistModel, width, color = 'r')
        bar2 = plt.bar(ind+width, verifiedModel, width, color='g')
        bar3 = plt.bar(ind+width*2, scrapedModel, width, color = 'b')

        plt.ylabel('Accuracy') 
        plt.title("Training and test scores")
        plt.xticks(ind+width, X)
        plt.legend( (bar1, bar2, bar3), ('mnist', 'verified', 'scraped') )
        plt.show()
        
        