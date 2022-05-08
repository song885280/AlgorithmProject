

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(3 * np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'], columns=['x'])
df.plot.pie(subplots=True)
plt.show()
