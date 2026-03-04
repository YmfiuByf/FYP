import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from preprocessing import *
from tqdm import tqdm
from torch.optim import *



def main(k,load_path,save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load pretrained model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    if k==1:
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    else:
        model = torch.load(load_path)
    model = model.to(device)
    signals=[]
    get_original_signal(signals,f'D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\sentences\\wav')
    texts = []
    get_text(texts, f"D:\\pycharmProject\\FYP\\IEMOCAP语料库\\Session{k}\\dialog\\transcriptions")

    X_train = signals
    y_train = texts
    def audio2label(audio_input,target_transcription):
        input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        # retrieve logits & take argmax
        with processor.as_target_processor():
            label = processor(target_transcription, return_tensors="pt").input_ids
        return input_values, label

    def train(model, X_train, y_train,loss_function, optimizer,scheduler,  epochs=150, batch_size=2,path='D:\\pycharmProject\\FYP'):
        model.train()
        for i in tqdm(range(epochs)):
            for b in range(len(X_train)//batch_size -1 ):
                x = X_train[b]
                input_values,label = audio2label(x,y_train[b])
                # print(y_pred.size(),label.size())
                loss = model(input_values, labels=label).loss
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f'epoch={i},loss={loss}')
            torch.save(model, path)
        return

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    train(model, X_train, y_train, None, optimizer, scheduler, epochs=30, batch_size=1,
          path=save_path)

if __name__=='__main__':
    k=2
    load_path = f"D:\\pycharmProject\\FYP\\ASR{k-1}.pth"
    save_path = f"D:\\pycharmProject\\FYP\\ASR{k}.pth"
    main(k,load_path,save_path)

