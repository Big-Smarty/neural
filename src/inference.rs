use burn::{
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::vision::MNISTItem},
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use crate::{data::MNISTBatcher, training::TrainingConfig};

pub fn infer_on_index<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");
    let model = config.model.init_with::<B>(record);

    let label = item.label;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} expected {}", predicted, label);
}

pub fn infer_on_file<B: Backend>(artifact_dir: &str, device: B::Device, image_path: &str) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");
    let model = config.model.init_with::<B>(record);

    let label = 2;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![{
        let img = image::imageops::flip_vertical(&image::imageops::rotate270(
            &image::open(image_path).unwrap().into_luma16(),
        ));
        img.save("test_img.png").unwrap();
        let mut matrice: [[f32; 28]; 28] = [[0.0; 28]; 28];
        for x in 0..27 {
            for y in 0..27 {
                matrice[x][y] = img.get_pixel(x as u32, y as u32)[0] as f32;
            }
        }
        MNISTItem {
            image: matrice,
            label,
        }
    }]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} expected {}", predicted, label);
}
