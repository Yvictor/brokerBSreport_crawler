import requests
import io
from bs4 import BeautifulSoup as BS
import pandas as pd
import sys
import time
from datetime import date,datetime,timedelta
import random
import numpy as np
from PIL import Image as IM
import pytesseract
import os
import cv2
import re
import asyncio
import hangups
from scipy.misc import toimage
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.utils import np_utils


def getint(x):
    if type(x) == str:
        return int(('').join(x.split(',')))
    else:
        return x
def divexpectz(a,b):
    if b == 0:
        return 0
    else:
        return round(a/b,2)
def returnstat(id0,reposttime):
    text = "\r {0}重新取得驗證碼次數:{1}".format(id0,reposttime)
    sys.stdout.write(text)
    sys.stdout.flush()

refpath = os.getcwd()+'/Notebooks/'
CONVERSATION_ID = 'Ugx10p7pgNA_uogOqn54AaABAagB3ZWGCA'
REFRESH_TOKEN_PATH = refpath+'refresh_token.txt'
    
def Hangouts():
    cookies = hangups.auth.get_auth_stdin(REFRESH_TOKEN_PATH)
    client = hangups.Client(cookies)
    client.on_connect.add_observer(lambda: asyncio.async(send_message(client)))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.connect())
@asyncio.coroutine
def send_message(client):
    """Send message using connected hangups.Client instance."""
    request = hangups.hangouts_pb2.SendChatMessageRequest(
        request_header=client.get_request_header(),
        event_request_header=hangups.hangouts_pb2.EventRequestHeader(
            conversation_id=hangups.hangouts_pb2.ConversationId(id=CONVERSATION_ID),
            client_generated_id=client.get_client_generated_id(),),
        message_content=hangups.hangouts_pb2.MessageContent(
            segment=[hangups.ChatMessageSegment(MESSAGE).serialize()],),)
    try:
        yield from client.send_chat_message(request)
    finally:
        yield from client.disconnect()

class captcha_recognize:
    def __init__(self):
        self.model = None
        self.lable = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
       'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    def one_hot_reverse(self,onehot):
        return self.lable[np.where(onehot==1)[0][0]]
    def load_model(self):
        self.model = model_from_json(open(refpath+'TPEX_cnn_captcha.json').read())
        self.model.load_weights(refpath+'TPEX_captcha_weights.h5')

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
    def preprocess(self,image):
        X = []
        imgpos = [[0,32],[25,57],[49,81],[72,104],[98,130]]
        #image = IM.open('/home/ubuntu/Notebooks/captest.png')
        imgarray = np.asarray(image.convert(mode='RGB'))
        #print(imgarray.shape)
        for tid in range(5):
            X.append(imgarray[:,imgpos[tid][0]:imgpos[tid][1]].reshape(3,32,30))
        X = np.array(X).astype('float32')
        X /= 255
        return X
    
    def captcha_predict(self,X):
        if type(self.model)!= Sequential:
            self.load_model()
        ans = self.model.predict(X)
        captcha =''
        for i in ans:
            captcha += self.lable[i.argmax()]
        return captcha
        
    
class TPEXBSreport:
    def __init__(self):
        self.rs = requests.session()
        self.curpath = os.getcwd()+'/Notebooks/'
        self.datenow = self.__getdate()
        self.notradedata = []
        stockidt = pd.read_csv(self.curpath+'csv_data/listcompanyo.csv',encoding='utf-8')
        self.stockidL = stockidt['股票代號'].tolist()
        self.captcha_rec = captcha_recognize()
    def __constr(self,l):
        an = re.sub('[^A-Z0-9]','',l.upper())
        if len(an) > 5:
            an = re.sub('[^A-HJ-Z0-9]','',l.upper())
        if '0' in an:
            an = an.replace('0','Q')
        if 'O' in an:
            an = an.replace('O','Q')
        return an
    def __getdate(self):
        d = datetime.now()
        if d.hour<16:
            d = d.date() - timedelta(1)
        if type(d) != type(date.today()):
                d = d.date()
        if d.isoweekday() == 7:
            d = d - timedelta(2)
        elif d.isoweekday() == 6:
            d = d - timedelta(1)
        return d
    def getCaptcha(self):
        sleeptime = 30
        captcha = None
        while str(captcha) != '<Response [200]>':
            try:
                captcha = self.rs.get('http://www.tpex.org.tw/web/inc/authnum.php',stream=True, verify=False)
            except:
                time.sleep(sleeptime)
                sleeptime+=300
        return captcha.content
    def cv2image(self):
        captchat_byte = self.getCaptcha()
        image_rgb = cv2.imdecode(np.frombuffer(captchat_byte,dtype=np.uint8),flags=1)
        image_gray = cv2.imdecode(np.frombuffer(captchat_byte,dtype=np.uint8),flags=0)
        cv2.imwrite(self.curpath+"ocrcorrect/otcoir/bkcapt.png", image_rgb)
        return image_gray
    def Captcha_preprocess(self):
        image = self.cv2image()
        im = cv2.resize(image, (260,60))
        retval, im = cv2.threshold(im,120, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(im,kernel,iterations = 1)
        blurred = cv2.GaussianBlur(erosion, (3, 3), 5)
        cv2.imwrite(self.curpath+"bkcapt.png", blurred)
        return 1
        
    def OCR(self):
        return self.captcha_rec.captcha_predict(self.captcha_rec.preprocess(IM.open(io.BytesIO(self.getCaptcha()))))
    
    def savecorimg(self, captcha):
        if os.path.exists(self.curpath+'ocrcorrect/otcoir/%s.png'%captcha)== False:
            os.rename(self.curpath+'ocrcorrect/otcoir/bkcapt.png',
                      self.curpath+'ocrcorrect/otcoir/%s.png'%captcha)
        elif os.path.exists(self.curpath+'ocrcorrect/otcoir/%s.png'%captcha)== True:
            nonew = 1
            npng = captcha+'_'+str(nonew)
            while os.path.exists(self.curpath+'ocrcorrect/otcoir/%s.png'%npng)== True:
                nonew +=1
                npng = captcha+'_'+str(nonew)
            if os.path.exists(self.curpath+'ocrcorrect/otcoir/%s.png'%npng)== False:
                os.rename(self.curpath+'ocrcorrect/otcoir/bkcapt.png',
                          self.curpath+'ocrcorrect/otcoir/%s.png'%npng)
            
    def postpayload(self,stockid,captcha,urltype):
        payload = {
            'stk_code': '%s'%str(stockid),
            'auth_num':captcha
        }
        res = self.rs.post('http://www.tpex.org.tw/web/stock/aftertrading/broker_trading/brokerBS.php',data = payload)
        res.encoding = 'utf-8'
        soup = BS(res.text,"lxml")
        self.answ = soup.select('.pt10')[0].text  
        if self.answ == '\n ***驗證碼錯誤，請重新查詢。*** \n':
            correctanswer = 0
        elif self.answ == '\n ***該股票該日無交易資訊*** \n':
            return 2
        elif self.answ[0:7] == '\n\n\n交易日期':
            correctanswer = 1

            self.ind = pd.read_html(str(soup.select('.table-striped')[0]))[0]
            self.dtda = re.sub('[^0-9]','/',self.ind[1][0]).split('/')[0:3]
            stkd = ('').join(self.dtda)
            urlbig5 = 'http://www.tpex.org.tw/web/stock/aftertrading/broker_trading/download_ALLCSV.php?curstk='
            urlutf8 = 'http://www.tpex.org.tw/web/stock/aftertrading/broker_trading/download_ALLCSV_UTF-8.php?curstk='
            if urltype == 5:
                url = urlbig5+str(stockid)+'&stk_date='+stkd+'&auth='+captcha
            elif urltype ==8:
                url = urlutf8+str(stockid)+'&stk_date='+stkd+'&auth='+captcha
            self.csvf = self.rs.get(url,stream=True, verify=False)
        else:
            correctanswer = 0
        return correctanswer
    
    def processdata(self,stockid):
        dat = date(int(self.dtda[0])+1911,int(self.dtda[1]),int(self.dtda[2]))
        tda = int(self.ind[1][1])
        ap = int(re.sub('[^0-9]','',self.ind[3][1]))
        allshare = self.ind[5][1]
        rt_ratio = self.ind[7][1]
        op = float(self.ind[1][2])
        hp = float(self.ind[3][2])
        lp = float(self.ind[5][2])
        cp = float(self.ind[7][2])
        d = {"日期":dat,
             "代號":stockid,
             "成交筆數":tda,
             "總成交金額":ap,
             "總成交股數":allshare,
             "周轉率(%)":rt_ratio,
             "開盤價":op,
             "最高價":hp,
             "最低價":lp,
             "收盤價":cp}
        ind = pd.DataFrame(d, index=[1])
        ind.index.name = '序號'
        tablens = pd.read_csv(io.StringIO(self.csvf.text.split('證券代碼')[1][7:]),encoding='utf-8')
        table00 = tablens[['序號','券商','價格','買進股數','賣出股數']]
        table01 = tablens[['序號.1','券商.1','價格.1','買進股數.1','賣出股數.1']]
        table00.columns = ['序號','證券商','成交單價','買進股數','賣出股數']
        table01.columns = ['序號','證券商','成交單價','買進股數','賣出股數']
        frame00 = [table00,table01]
        table = pd.concat(frame00)
        table = table.set_index('序號')
        table = table.dropna()
        table = table.sort_index()
        table.index.name = '序'
        table['買進股數'] = table['買進股數'].map(lambda x: getint(x))
        table['賣出股數'] = table['賣出股數'].map(lambda x: getint(x))
        table = table.join(ind)
        table[["日期","代號","成交筆數","總成交金額","總成交股數","周轉率(%)","開盤價","最高價","最低價","收盤價"]] = table[["日期","代號","成交筆數","總成交金額","總成交股數","周轉率(%)","開盤價","最高價","最低價","收盤價"]].fillna(method='pad')
        table = table[["日期","代號","成交筆數","總成交金額","總成交股數","周轉率(%)","開盤價","最高價","最低價","收盤價","證券商","成交單價","買進股數","賣出股數"]]
        table['日期'] = pd.to_datetime(table['日期'])
        filename = str(stockid)+"_"+('').join(str(dat).split('-'))
        table.to_csv(self.curpath+'csv_data/stockdt/ori/%s.csv'%filename)
        #######################################################################################
        buyp = table.apply(lambda row: row['成交單價']*row['買進股數'],axis=1)
        table.insert(13,'買進金額',buyp)
        sellp = table.apply(lambda row: row['成交單價']*row['賣出股數'],axis=1)
        table.insert(14,'賣出金額',sellp)
        table_sort = table.groupby(["日期","代號","成交筆數","總成交金額","總成交股數","周轉率(%)","開盤價","最高價","最低價","收盤價","證券商"])[['買進股數','賣出股數','買進金額','賣出金額']].sum()
        table_sort = table_sort.reset_index(["成交筆數","總成交金額","總成交股數","周轉率(%)","開盤價","最高價","最低價","收盤價"])
        table_sort = table_sort[['買進股數','賣出股數','買進金額','賣出金額',"成交筆數","總成交金額","總成交股數","周轉率(%)","開盤價","最高價","最低價","收盤價"]]
        b_avg_p = table_sort.apply(lambda row: divexpectz(row['買進金額'],row['買進股數']),axis=1)
        s_avg_p = table_sort.apply(lambda row: divexpectz(row['賣出金額'],row['賣出股數']),axis=1)
        b_ratio = table_sort.apply(lambda row: divexpectz(row['買進股數'],row['總成交股數'])*100,axis=1)
        s_ratio = table_sort.apply(lambda row: divexpectz(row['賣出股數'],row['總成交股數'])*100,axis=1)
        bs_share_net = table_sort.apply(lambda row: row['買進股數']-row['賣出股數'],axis=1)
        bs_price_net = table_sort.apply(lambda row: row['買進金額']-row['賣出金額'],axis=1)
        table_sort.insert(2,'買賣超股數',bs_share_net)
        table_sort.insert(5,'買賣超金額',bs_price_net)
        table_sort.insert(6,'買進均價',b_avg_p)
        table_sort.insert(7,'賣出均價',s_avg_p)
        table_sort.insert(8,'買進比重',b_ratio)
        table_sort.insert(9,'賣出比重',s_ratio)
        if os.path.exists(self.curpath+'csv_data/stockdt/sort/%s.csv'%str(stockid)) == False:
            table_sort.to_csv(self.curpath+'csv_data/stockdt/sort/%s.csv'%str(stockid))
        #after first time use
        table_s = pd.read_csv(self.curpath+'csv_data/stockdt/sort/%s.csv'%str(stockid),encoding='utf-8',index_col=[0,1,2],parse_dates=[0])
        if dat not in table_s.index.levels[0]:
            frame = [table_sort,table_s]
            table_s = pd.concat(frame)
        table_s.to_csv(self.curpath+'csv_data/stockdt/sort/%s.csv'%str(stockid))
    
    def singleprocess(self,stockid):
        anscor = 0
        repostcount = 0
        changeurltype = 0
        filename = str(stockid)+"_"+('').join(str(self.datenow).split('-'))
        if os.path.exists(self.curpath+'csv_data/stockdt/ori/%s.csv'%filename) == False:
            while anscor == 0:
                if changeurltype ==0:
                    urltype = 5
                elif changeurltype==1:
                    urltype = 8
                Capt = 0
                while Capt ==0:
                    try:
                        Capt = self.OCR()
                    except:
                        pass
                anscor = self.postpayload(stockid, Capt, urltype)
                returnstat(stockid,repostcount)
                repostcount +=1
                if anscor == 2:
                    self.notradedata.append(stockid)
                    break
                if repostcount>150:
                    repostcount = 150
                    break
                time.sleep(random.choice([2.8,3.2,3.8,4.1,4.7]))
                while anscor == 1 and int(self.dtda[0])<85:
                    anscor = self.postpayload(stockid, Capt, urltype)
                    time.sleep(random.choice([1.3,1.8,1.4,1.1,1.5]))
                if anscor == 1 and int(self.dtda[0])>85:
                    try:
                        self.processdata(stockid)
                        changeurltype = 0
                    except:
                        changeurltype = 1
                        anscor = 0
        if os.path.exists(self.curpath+'csv_data/stockdt/ori/%s.csv'%filename) == True:
            repostcount = 100
        return stockid,repostcount
    
    def processAll(self):
        starttime = datetime.now()
        stlen = len(self.stockidL)
        self.arrcu = []
        global MESSAGE
        MESSAGE = ''
        for i in range(stlen):
            a = self.singleprocess(self.stockidL[i])
            if a[1]==150:
                a = self.singleprocess(self.stockidL[i])
            ptime = datetime.now()
            text = "\r上櫃 {0}/{1} 已完成 {2}%  處理時間: {3}".format(i+1,stlen,round((i+1)/stlen,4)*100,str(ptime-starttime))
            sys.stdout.write(text)
            sys.stdout.flush()
            if i%200 == 0:
                MESSAGE = "上櫃 {0}/{1} 已完成 {2}%  處理時間: {3}".format(i+1,stlen,round((i+1)/stlen,4)*100,str(ptime-starttime))
                Hangouts()
            self.arrcu.append(a)
            if self.arrcu[-1][1] == 0:
                time.sleep(3)
            if len(self.arrcu)>3 and self.arrcu[-1][1] == 0 and self.arrcu[-2][1] == 0 and self.arrcu[-3][1] == 0:
                time.sleep(5)
        endtime = datetime.now()
        spendt = str(endtime - starttime)
        MESSAGE = "上櫃股票交易日報下載完成 \n 花費時間:{0}".format(spendt)
        Hangouts()
    
    def checkunprocesslist(self):
        datestr = ('').join(str(self.datenow).split('-'))+'.csv'
        fprocessed = [int(i.split('_')[0]) for i in os.listdir(self.curpath+'csv_data/stockdt/ori') if i.endswith(datestr)==True] + self.notradedata
        self.unprocess = [i for i in self.stockidL if i not in fprocessed]
        for i in self.unprocess:
            a = self.singleprocess(i)
        text = str((round(len(self.unprocess)/len(self.stockidL),4)*100,len(self.unprocess),len(self.stockidL)))
        MESSAGE = "上櫃股票交易日報檢查完畢 \n {0}".format(text)
        Hangouts()
        return round(len(self.unprocess)/len(self.stockidL),4)*100,len(self.unprocess),len(self.stockidL)

        
BSr = TPEXBSreport()
BSr.processAll()
BSr.checkunprocesslist()