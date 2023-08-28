from flask import Flask,render_template,request
import pickle
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("indexyo.html")

@app.route("/predict",methods=["GET","POST"])
def pre():
    name=request.form["Name"]
    ssc=request.form["ssc"]
    hsc=request.form["hsc"]
    degree=request.form["Degreeper"]
    test=request.form["test"]
    mba=request.form["mba"]
    ssc=float(ssc)
    hsc=float(hsc)
    test=float(test)
    mba=float(mba)
    degree=float(degree)
    gender=int(request.form["Gender"])
    dtype=int(request.form["Degree"])
    spe=int(request.form["specialisation"])
    wexp=int(request.form["Work"])
    sci=0
    com=0
    if(dtype==1) :
        sci=1
        com=0
    elif(dtype==2):
        sci=0
        com=1
    arr=[[gender,ssc,hsc,degree,wexp,test,spe,mba,com,sci]]

    with open('LogisticReg','rb') as f:
        m1 = pickle.load(f)
    
    with open('RandomForest','rb') as f:
        m2 = pickle.load(f)

    with open('DecisionTree','rb') as f:
        m3 = pickle.load(f)
    l1=m1.predict(arr)
    l2=m2.predict(arr)
    l3=m3.predict(arr)

    sum=(l1[0]+l2[0]+l3[0])
    if(sum==0):
        return render_template("yo1.html",name=name,result="Your chances of getting placement are  low")
        # return f'{name} 
    elif(sum==1):
        return render_template("yo1.html",name=name,result="Your chances of getting placement are Moderate")
    elif(sum==2):
        return render_template("yo1.html",name=name,result="Your chances of getting placement are High")
    else:
        return render_template("yo1.html",name=name,result="Your chances of getting placement are very High")









if __name__=="__main__" :
    app.run(debug=True)
