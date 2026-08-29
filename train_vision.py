from ultralytics import YOLO

def train_compliance_model():
    print("Initializing YOLOv8 Nano...")
    model = YOLO("yolov8n.pt") 

    print("Beginning hardware-accelerated training...")
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,          
        imgsz=640,          
        batch=16,           
        device=0,           
        project="models",   
        name="legal_vision" 
    )
    
    print("Training complete! Best model saved to: models/legal_vision/weights/best.pt")

if __name__ == "__main__":
    train_compliance_model()