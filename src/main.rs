mod data;
mod inference;
mod model;
mod training;

use std::path::PathBuf;

use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::data::dataset::vision::MNISTDataset;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use model::ModelConfig;

use clap::{arg, command, value_parser, Command};

fn main() {
    let matches = command!()
        .subcommand(
            Command::new("infer")
                .about("Infers the model")
                .arg(
                    arg!(-i --index <Index> "index of the MNIST dataset to infer on")
                        .required(false)
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    arg!(-f --file <File> "custom file to infer the network on")
                        .required(false)
                        .value_parser(value_parser!(PathBuf)),
                ),
        )
        .subcommand(Command::new("train").about("Trains the model"))
        .get_matches();
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::BestAvailable;

    if let Some(matches) = matches.subcommand_matches("infer") {
        if let Some(file) = matches.get_one::<PathBuf>("file") {
            inference::infer_on_file::<MyBackend>("guide", device.clone(), file.to_str().unwrap());
        } else if let Some(index) = matches.get_one::<usize>("index") {
            inference::infer_on_index::<MyBackend>(
                "guide",
                device.clone(),
                MNISTDataset::test().get(*index).unwrap(),
            );
        } else {
            inference::infer_on_index::<MyBackend>(
                "guide",
                device.clone(),
                MNISTDataset::test().get(42).unwrap(),
            );
        }
    }

    if matches.subcommand_matches("train").is_some() {
        training::train::<MyAutodiffBackend>(
            "guide",
            training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
            device.clone(),
        );
    }
}
