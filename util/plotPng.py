import matplotlib.pyplot as plt

def plotPng(loss_plot,acc_plot):
    plt.style.use(['science','no-latex','cjk-sc-font'])
    fig = plt.figure(figsize=(16, 12), dpi=80)
    plt.title("Training loss and accuracy") 
    plt.xlabel("train epoch") 
    plt.ylabel("acc or loss") 
    plt.plot(loss_plot, color='firebrick',   label='training loss')
    plt.plot(acc_plot,  color='darkorange',   label='accuracy')
    plt.legend() 
    plt.savefig("Path_to_store_your_png")