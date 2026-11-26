from pathlib import Path
import os, sys, io, json, tempfile, shutil, subprocess, threading, zipfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(os.environ.get('TRAFFICGUARD_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / 'src'))
st.set_page_config(page_title='TrafficGuard AI', page_icon='🛡️', layout='wide')
st.markdown('''<style>.block-container{max-width:1250px;padding-top:2rem} div[data-testid="stMetric"]{border:1px solid #64748b55;border-radius:12px;padding:14px} </style>''', unsafe_allow_html=True)
MODES = ['combined','helmet','seatbelt']
PROFILES = ['standard','high_resolution','high_sensitivity','region_assisted','maximum_resolution']
EXT = ['jpg','jpeg','png','bmp','webp']
COUNTS = ['with_helmet','without_helmet','with_seatbelt','without_seatbelt']

@st.cache_resource
def load_engine(root):
    from trafficguard_refined import RefinedTrafficGuard
    return RefinedTrafficGuard(root, profile='standard'), threading.RLock()

def get_engine():
    return load_engine(str(ROOT))

def decode(data):
    image = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Invalid or unreadable image')
    return image

def infer(image,mode,profile):
    engine, lock = get_engine()
    with lock:
        engine.set_profile(profile)
        return engine.process_image(image,mode=mode,use_context=True)

def encode(image):
    ok, buf = cv2.imencode('.png',image)
    if not ok: raise ValueError('Image encoding failed')
    return buf.tobytes()

def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def summary_cards(s):
    cols=st.columns(4)
    for col,label,key in zip(cols,['Raw detections','Possible helmet violations','Possible seatbelt violations','Context unverified'],['total_detections','possible_helmet_violations','possible_seatbelt_violations','unverified_detections']):
        col.metric(label,int(s.get(key,0)))
    st.caption('Counts are model predictions, not confirmed violations. Zero detections does not establish compliance.')

def show_image_result(item,i):
    st.subheader(item['name'])
    a,b=st.columns(2)
    a.image(item['original'],caption='Original image')
    b.image(item['annotated'],caption='Annotated prediction')
    summary_cards(item['summary'])
    st.dataframe(pd.DataFrame(item['detections']),hide_index=True)
    st.caption(f"Profile: {item['summary']['profile']} · Processing: {item['summary']['processing_ms']:.0f} ms")
    st.download_button('Download annotated image',item['annotated'],f'{Path(item["name"]).stem}_detected.png','image/png',key=f'img{i}')

def save_image_result(name,data,mode,profile):
    image=decode(data); r=infer(image,mode,profile)
    return {'name':name,'original':encode(image),'annotated':encode(r['annotated_image']),'summary':r['summary'],'detections':r['detections']}

st.sidebar.title('TrafficGuard AI')
page=st.sidebar.radio('Navigation',['Overview','Image detection','Video detection','CSV batch','Model performance'])
mode=st.sidebar.selectbox('Detection mode',MODES)
profile=st.sidebar.selectbox('Inference profile',PROFILES)
st.sidebar.caption('Standard is the evaluated default; other profiles are experimental.')
st.sidebar.caption('Helmet 0.30 / Seatbelt 0.35 in standard mode. Human review required.')
st.title('TrafficGuard AI')
st.caption('Computer vision for traffic safety • Academic prototype')

if page=='Overview':
    st.header('Traffic safety monitoring')
    a,b=st.columns(2); a.metric('Helmet test mAP50','72.22%'); b.metric('Seatbelt test mAP50','92.93%')
    st.markdown('Upload traffic images, analyze sampled video frames, or process a CSV manifest of uploaded images. The system combines two custom YOLO models with a pretrained vehicle-context detector.')
    st.info('Green: context-supported compliant prediction. Red: possible violation. Orange: context unverified. No automatic enforcement.')
    st.warning('Distant, blurred or obstructed traffic footage may yield missed detections. Video counts are per-frame instances, not unique people.')
    if st.button('Verify and load models'):
        try:
            eng,_=get_engine(); st.success(f'All three models loaded and verified. Device: {eng.device}')
        except Exception as e: st.error(str(e))

elif page=='Image detection':
    st.header('Image detection')
    uploads=st.file_uploader('Upload traffic images',type=EXT,accept_multiple_files=True,key='images')
    if st.button('Run image detection',type='primary',disabled=not uploads):
        st.session_state['image_results']=[]
        for f in uploads:
            try:
                with st.spinner(f'Processing {f.name}'):
                    st.session_state['image_results'].append(save_image_result(f.name,f.getvalue(),mode,profile))
            except Exception as e: st.error(f'{f.name}: {e}')
    items=st.session_state.get('image_results',[])
    for i,item in enumerate(items): show_image_result(item,i)
    if items:
        df=pd.DataFrame([{'image':r['name'],**r['summary']} for r in items])
        st.download_button('Download image summary CSV',csv_bytes(df),'image_summary.csv','text/csv')
        st.download_button('Download detailed detections JSON',json.dumps([{ 'image':r['name'],'detections':r['detections']} for r in items],indent=2),'detections.json','application/json')

elif page=='Video detection':
    st.header('Video detection')
    st.warning('Use clear, close-range traffic footage. CPU processing can be slow. Output contains sampled frames and no audio.')
    f=st.file_uploader('Upload video',type=['mp4','avi','mov','mkv','webm'],key='video')
    target=st.slider('Target sampled frames per second',1,10,2)
    if f: st.video(f.getvalue())
    if st.button('Process video',type='primary',disabled=f is None):
        st.session_state.pop('video_result',None)
        try:
            from video_processor import process_video
            with tempfile.TemporaryDirectory() as tmp:
                src=Path(tmp)/('input'+Path(f.name).suffix.lower()); src.write_bytes(f.getvalue())
                cap=cv2.VideoCapture(str(src)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
                if not np.isfinite(fps) or fps<=0: raise ValueError('Invalid video FPS')
                stride=max(1,round(fps/target)); out=Path(tmp)/'output.mp4'
                bar=st.progress(0)
                def progress(n,total):
                    if total>0: bar.progress(min(1.0,n/total))
                eng,lock=get_engine()
                with lock:
                    eng.set_profile(profile)
                    report=process_video(eng,src,out,mode=mode,frame_stride=stride,max_frames=None,progress_callback=progress)
                report.update(profile=profile,mode=mode,original_filename=f.name)
                playable=out; browser_ready=False
                if shutil.which('ffmpeg'):
                    converted=Path(tmp)/'playable.mp4'
                    p=subprocess.run(['ffmpeg','-y','-i',str(out),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(converted)],capture_output=True)
                    if p.returncode==0: playable=converted; browser_ready=True
                st.session_state['video_result']={'video':playable.read_bytes(),'csv':out.with_suffix('.csv').read_bytes(),'report':report,'browser_ready':browser_ready}
                bar.progress(1.0)
        except Exception as e: st.error(str(e))
    r=st.session_state.get('video_result')
    if r:
        if r['browser_ready']: st.video(r['video'])
        else: st.info('Browser-compatible conversion unavailable; download the MP4 to view locally.')
        st.json(r['report'])
        df=pd.read_csv(io.BytesIO(r['csv'])); st.dataframe(df,hide_index=True)
        cols=[c for c in ['possible_helmet_violations','possible_seatbelt_violations','unverified_detections','total_detections'] if c in df]
        if cols: st.line_chart(df.set_index('timestamp_seconds')[cols])
        st.download_button('Download annotated video',r['video'],'trafficguard_output.mp4','video/mp4')
        st.download_button('Download frame CSV',r['csv'],'frame_statistics.csv','text/csv')
        st.download_button('Download video report',json.dumps(r['report'],indent=2),'video_report.json','application/json')

elif page=='CSV batch':
    st.header('CSV batch detection')
    st.write('Upload a CSV manifest and the referenced images together. Use exact filenames in image_path. Optional expected counts compare raw detections only.')
    template=pd.DataFrame([{'image_path':'rider.jpg','mode':'helmet','expected_with_helmet':0,'expected_without_helmet':2},{'image_path':'car.jpg','mode':'seatbelt','expected_with_seatbelt':1,'expected_without_seatbelt':0}])
    st.download_button('Download CSV template',csv_bytes(template),'batch_template.csv','text/csv')
    manifest=st.file_uploader('CSV manifest',type=['csv'],key='manifest')
    images=st.file_uploader('Referenced images',type=EXT,accept_multiple_files=True,key='batch_images')
    if st.button('Run batch',type='primary',disabled=not manifest or not images):
        st.session_state.pop('batch_result',None)
        try:
            df=pd.read_csv(io.BytesIO(manifest.getvalue()),encoding='utf-8-sig'); df.columns=df.columns.str.strip()
            if 'image_path' not in df or df.empty: raise ValueError('CSV requires a nonempty image_path column')
            if len(df)>100: raise ValueError('Maximum 100 rows per batch')
            lookup={}
            for f in images:
                if f.name in lookup: raise ValueError(f'Duplicate uploaded filename: {f.name}')
                lookup[f.name]=f.getvalue()
            records=[]; buf=io.BytesIO(); progress=st.progress(0)
            with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                for i,row in df.iterrows():
                    name=str(row['image_path']).strip(); selected=row.get('mode',mode); selected=mode if pd.isna(selected) else str(selected).strip().lower()
                    try:
                        if name not in lookup: raise ValueError('Referenced image not uploaded')
                        if selected not in MODES: raise ValueError('Invalid mode')
                        expected={}
                        for key in COUNTS:
                            value=row.get('expected_'+key)
                            if pd.notna(value):
                                value=float(value)
                                if not np.isfinite(value) or value<0 or not value.is_integer(): raise ValueError('Expected counts must be nonnegative integers')
                                if selected=='combined' or key.endswith(selected): expected[key]=int(value)
                        item=save_image_result(name,lookup[name],selected,profile)
                        record={'row':i,'image':name,'status':'PROCESSED',**item['summary']}
                        for key,value in expected.items(): record['expected_'+key]=value; record['difference_'+key]=item['summary'][key]-value
                        record['count_check']='MATCH' if expected and all(item['summary'][k]==v for k,v in expected.items()) else 'MISMATCH' if expected else 'NOT PROVIDED'
                        z.writestr(f'{i:04d}_detected.png',item['annotated'])
                        z.writestr(f'{i:04d}_detections.json',json.dumps(item['detections'],indent=2))
                    except Exception as e: record={'row':i,'image':name,'status':'ERROR','error':str(e)}
                    records.append(record); progress.progress((i+1)/len(df))
                result=pd.DataFrame(records); data=csv_bytes(result); z.writestr('batch_results.csv',data)
            st.session_state['batch_result']={'csv':data,'zip':buf.getvalue()}
        except Exception as e: st.error(str(e))
    r=st.session_state.get('batch_result')
    if r:
        df=pd.read_csv(io.BytesIO(r['csv'])); st.dataframe(df,hide_index=True)
        st.metric('Successfully processed rows',int((df['status']=='PROCESSED').sum()))
        st.caption('Count matching is not localization accuracy or unique violation counting.')
        st.download_button('Download batch results CSV',r['csv'],'batch_results.csv','text/csv')
        st.download_button('Download annotated batch ZIP',r['zip'],'batch_outputs.zip','application/zip')

elif page=='Model performance':
    st.header('Held-out model evaluation')
    path=ROOT/'results/final_evaluation/tables/independent_test_metrics.csv'
    if path.is_file():
        df=pd.read_csv(path); metrics=[c for c in ['Precision','Recall','Macro F1','mAP50','mAP50-95','Violation F1'] if c in df]
        shown=df.copy()
        for c in metrics: shown[c]=shown[c]*100
        st.dataframe(shown.round(2),hide_index=True)
        if metrics and 'Task' in shown: st.bar_chart(shown.set_index('Task')[metrics].T)
        st.download_button('Download original evaluation CSV',path.read_bytes(),path.name,'text/csv')
    else: st.warning('Original evaluation CSV not found. Restore it from the verified Phase 7–8 backup.')
    st.info('These metrics describe the original custom detectors, not the context-assisted combined pipeline. Diagnostic samples are not independent tests.')
    st.write('Helmet: mixed rider and cyclist annotations; limited localization and generalization. Seatbelt: visibility through windshield is required. External quantitative testing remains pending.')
