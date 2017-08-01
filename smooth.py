import numpy as np
import ipdb
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.interpolate import spline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
#cost =  np.genfromtxt("cost.csv", delimiter = ',')
#costr = np.genfromtxt("costr.csv", delimiter = ',')
#accuracy_10 = np.genfromtxt("accuracy_10.csv",delimiter = ',')
#accuracy_2 = np.genfromtxt("accuracys_10.csv",delimiter = ',')
#accuracyr_10 = np.genfromtxt("accuracyr_10.csv",delimiter = ',')
#accuracyr_2 = np.genfromtxt("accuracyrs_10.csv",delimiter = ',')
b5a1 = np.genfromtxt("cosm1.csv",delimiter = ',')
b5a2 = np.genfromtxt("cosm2.csv",delimiter = ',')
b5a5 = np.genfromtxt("bal10.csv",delimiter = ',')
b5a10 = np.genfromtxt("sof10.csv",delimiter = ',')


#accuracy_5 = np.genfromtxt("accuracy_5.csv",delimiter = ',')
#accuracy_2 = np.genfromtxt("accuracy_2.csv",delimiter = ',')
#accuracy_1 = np.genfromtxt("accuracy_1.csv",delimiter = ',')
def make_cute_graph(episode,accuracy, filter = False):
    shape = episode.shape[0]
    #e3 = episode[:1
    e1 = episode[:shape/5]
    e2 = episode[shape/5:]
    a1 = accuracy[:shape/5]
    a2 = accuracy[shape/5:]
    ef2 = np.empty((e2.shape[0]/16, 1))
    af2 = np.empty((a2.shape[0]/16, 1))
    if filter:
        af1 = np.empty((a1.shape[0]/5, 1))
        ef1 = np.empty((e1.shape[0]/5, 1))
        for i in range(e1.shape[0]):
            if (i % 5 == 0):
                ef1[i/5] = e1[i]
                af1[i/5] = a1[i]
    else:
        ef1 = e1.reshape(-1,1)
        af1 = a1.reshape(-1,1)
    for j in range(e2.shape[0]):
        if (j % 16 == 0):
            ef2[j/16] = e2[j]
            af2[j/16] = a2[j]
    fil_episode = np.concatenate((ef1,ef2), axis = 0)
    fil_accuracy = np.concatenate((af1,af2), axis = 0)
    return fil_episode.reshape(-1), fil_accuracy.reshape(-1)
def make_cute_graph2(episode, accuracy):
    filtered_step = np.empty((100,1))
    filtered_accuracy = np.empty((100,1))
    for i in range(episode.shape[0]):
        if (i %10 == 0):
            filtered_step[i/10] = episode[i]
            filtered_accuracy[i/10] = accuracy[i]
    return filtered_step,filtered_accuracy

#filtered_step = np.empty((250,1))
#for i in range(step.shape[0]):
 #   if (i % 4) == 0:
  #      filtered_step[i/4] = step[i]
step = b5a1[1:,1]
#step = accuracy_2[1:,1]
#filtered_step = np.empty((250,1))
#for i in range(step.shape[0]):
 #   if (i % 4) == 0:
  #      filtered_step[i/4] = step[i]
#print(step[-1])
#step = cost[1:,1]
#cost_val = cost[1:,2]
#costr_val = costr[1:,2]
#valr_2 = accuracyr_2[1:,2]
#print(valr_2.shape)
#valr_10 = accuracyr_10[1:,2]
#val_10 = accuracy_10[1:,2]
#val_2 = accuracy_2[1:,2]
#val5 = accuracy_5[1:,2]
#val2 = accuracy_2[1:,2]
b5a1_edit = b5a1[1:,2]
b5a2_edit = b5a2[1:,2]
b5a5_edit = b5a5[1:,2]
b5a10_edit = b5a10[1:,2]

#cost_smooth = savgol_filter(cost_val, 201,2)
#costr_smooth = savgol_filter(costr_val,201,2)
#val2_smooth = savgol_filter(val_2, 101,2)
#print(val2_smooth.shape)
#val10_smooth = savgol_filter(val_10, 101,2)
#valr10_smooth = savgol_filter(valr_10,101,2)
#valr2_smooth = savgol_filter(valr_2,101,2)
b5a1s = savgol_filter(b5a1_edit,101,2)
b5a2s = savgol_filter(b5a2_edit,201,2)
b5a5s = savgol_filter(b5a5_edit,101,2)
b5a10s = savgol_filter(b5a10_edit,101,2)

s1,a1 = make_cute_graph(step,b5a1s,True)
s2,a2 = make_cute_graph(step,b5a2s,True)
s5,a5 = make_cute_graph(step,b5a5s,True)
s10,a10 = make_cute_graph(step,b5a10s,True)


#print(a.shape)
#print(b.shape)
#filtered = np.empty((250,1))
#for i in range(val2_smooth.shape[0]):
 #       if (i % 4) == 0:
#            filtered[i/4] = val2_smooth[i]
#print(filtered[-1])
#print(val2_smooth[-1])

#val5_smooth = savgol_filter(val5,201,2)
#val2_smooth = savgol_filter(val2,201,2)
#val1_smooth = savgol_filter(val1,201,2)
#print(val.min())
#step_smooth = np.linspace(step.min(),step.max(), 300)
#ips = interp1d(step,val,kind = 'linear')
#val_smooth = spline(ips(step_smooth),101,3)

#plt.plot(step,cost_smooth,'ko')
#plt.plot(step,costr_smooth,'b+')
#plt.plot(step,valr2_smooth,label = "New_MANN 10th instance")
#plt.plot(step,valr10_smooth,label = "New_MANN 10th instance")
#plt.plot(step,val10_smooth, label = "MANN 10th instance")
#plt.plot(step,val2_smooth,label = "MANN 10th instance")
fig, ax = plt.subplots(figsize=(5.5, 4))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
loc = plticker.MultipleLocator(base=0.2) # this locator puts ticks at regular intervals
ax.yaxis.set_major_locator(loc)

plot1, = ax.plot(s1,a1,'o', color = '#F41E68',ls = '-', lw = 0.2, ms = 4,label = "1st" + "\n" +  "instance")
#plot1, = ax.plot(s1,a1,'mo',ls = '-', lw = 0.2, ms = 4,label = "1st" + "\n" +  "instance")
#plot2, = ax.plot(s2,a2,'o', color = '#E62B83',ls = '-', lw = 0.2, ms = 4,label = "2nd" + "\n" +  "instance")
plot2, = ax.plot(s2,a2,'o', color = '#4F9E87',ls = '-', lw = 0.2, ms = 4,label = "2nd" + "\n" +  "instance")
#plot3, = ax.plot(s5,a5,'o',ls = '-', color = '#3C62D8',lw = 0.2, ms = 4,label = "5th" + "\n" +  "instance")
#plot3, = ax.plot(s5,a5,'o',ls = '-', color = '#F41E68',lw = 0.2, ms = 4,label = "5th" + "\n" +  "instance")
#plot3, = ax.plot(s5,a5,'o',ls = '-', color = '#3C62D8',lw = 0.2, ms = 4,label = "5th" + "\n" +  "instance")
#plot4, = ax.plot(s10,a10,'o', color = '#17B98A',ls = '-', lw = 0.2, ms = 4,label = "10th" + "\n" +" instance")
#plt.legend([plot1, plot2,plot3,plot4], ["1st\ninstance", "2nd\ninstance","5th\ninstance", "10th\ninstance"],columnspacing=1,bbox_to_anchor=[0.5-0.01, 1.1+0.015], frameon = False, loc='upper center',fancybox = False, ncol=4)
#plt.legend([plot1, plot2], ["MANN 's 2nd\ninstance","SMNN's 2nd\ninstance"],columnspacing=1,bbox_to_anchor=[0.5-0.01, 1.1+0.015], frameon = False, loc='upper center',fancybox = False, ncol=2)
plt.legend([plot1, plot2], ["MANN","SMNN"],columnspacing=1,bbox_to_anchor=[0.5-0.01, 1.1+0.015], frameon = False, loc='upper center',fancybox = False, ncol=2)
# Hide the right and top spines
#plt.spines['right'].set_visible(False)
#plt.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
#plt.yaxis.set_ticks_position('left')
#plt.xaxis.set_ticks_position('bottom')
#l1 = plt.legend(handles = [plot1],loc = 2, bbox_to_anchor=(0.1-0.015, 1))
#plt.legend(handles = [plot2],loc = 'best', bbox_to_anchor=(0.3-0.015, 1))
#plt.scatter(a,b)
plt.xlabel("Episode")
plt.ylabel("Error rate")
#plt.ylabel("Percent Accuracy")
#plt.plot(step,val5_smooth)
#plt.plot(step,val2_smooth)
#plt.plot(step,val1_smooth)
#plt.plot(step,val)
#plt.legend()
plt.grid(True)
plt.show()
