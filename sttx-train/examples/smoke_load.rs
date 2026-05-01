use anyhow::Result;
use candle_core::{DType, Device};
use sttx_train::model;

#[tokio::main]
async fn main() -> Result<()> {
    let repo = std::env::var("STTX_MODEL_REPO").unwrap_or_else(|_| model::DEFAULT_MODEL_REPO.to_string());
    eprintln!("smoke_load: downloading + loading {repo}");

    let device = Device::Cpu;
    let dtype = DType::BF16;
    let m = model::load(&repo, device.clone(), dtype).await?;
    eprintln!(
        "loaded: vocab={} hidden={} layers={} head={}",
        m.config.vocab_size, m.config.hidden_size, m.config.num_hidden_layers, m.config.head_size
    );

    let mut state = model::fresh_state(&m.config, &device)?;
    let logits = m.model.forward_seq(&[1u32], &mut state)?;
    eprintln!("logits shape: {:?}", logits.dims());
    let v = model::final_logits_to_vec(&logits)?;
    eprintln!("first logit values: {:?}", &v[..5.min(v.len())]);
    println!(
        "{{\"vocab\":{},\"hidden\":{},\"layers\":{},\"logits_dims\":{:?},\"first_logits\":{:?}}}",
        m.config.vocab_size,
        m.config.hidden_size,
        m.config.num_hidden_layers,
        logits.dims(),
        &v[..5.min(v.len())]
    );
    Ok(())
}
