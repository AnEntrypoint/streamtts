use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::{DType, Device};
use sttx_core::trace::Trace;
use sttx_train::checkpoint;
use sttx_train::model::{self, LoadedModel};
use sttx_train::train::{TrainConfig, Trainer};
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer,
};

static FAILS: AtomicUsize = AtomicUsize::new(0);

fn check(label: &str, ok: bool) {
    if !ok {
        FAILS.fetch_add(1, Ordering::SeqCst);
        eprintln!("FAIL {label}");
    } else {
        println!("ok   {label}");
    }
}

fn build_byte_tokenizer(vocab_size: usize) -> Tokenizer {
    let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
    for i in 0..vocab_size.min(256) {
        let s = format!("\\x{:02x}", i);
        vocab.insert(s, i as u32);
    }
    vocab.insert("[UNK]".to_string(), (vocab_size - 1) as u32);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .expect("wordlevel");
    let mut tk = Tokenizer::new(model);
    tk.with_pre_tokenizer(Some(Whitespace {}));
    tk
}

#[test]
fn random_init_train_witness() {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let cfg = model::build_tiny_config();
    let st_path = std::env::temp_dir().join("sttx-tiny-rwkv7.safetensors");
    model::write_random_safetensors(&cfg, &st_path, 13).expect("write safetensors");
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[st_path.clone()], dtype, &device)
            .expect("mmap safetensors")
    };
    let rwkv_model =
        candle_transformers::models::rwkv_v7::Model::new(&cfg, vb).expect("Model::new");

    let mut state = model::fresh_state(&cfg, &device).expect("fresh_state");
    let logits = rwkv_model
        .forward_seq(&[1u32, 2, 3], &mut state)
        .expect("forward_seq");
    check("model_logits_shape", logits.dims() == &[cfg.vocab_size]);
    check("state_per_layer_count", state.per_layer.len() == cfg.num_hidden_layers);
    check(
        "state_kv_shape",
        state.per_layer[0].att_kv.dims()
            == &[cfg.hidden_size / cfg.head_size, cfg.head_size, cfg.head_size],
    );

    let tokenizer = build_byte_tokenizer(cfg.vocab_size);
    let loaded = LoadedModel {
        model: rwkv_model,
        config: cfg,
        tokenizer,
        device: device.clone(),
        dtype,
    };

    let train_cfg = TrainConfig {
        steps: 10,
        max_tokens_per_step: 16,
        replay_capacity: 8,
        replay_per_step: 1,
        replay_weight: 0.1,
        lr: 5e-3,
        checkpoint_dir: PathBuf::from(std::env::temp_dir().join("sttx-train-witness")),
        checkpoint_every: 100,
        seed: 7,
    };
    let mut trainer = Trainer::new(loaded, train_cfg).expect("new trainer");

    let traces = [
        "abc abc abc abc abc abc abc abc",
        "the the the the the the",
        "hello hello hello hello hello",
        "code code code code code",
        "data data data data data",
    ];
    let mut losses = Vec::new();
    for (i, txt) in traces.iter().cycle().take(8).enumerate() {
        let trace = Trace::UserMessage { text: (*txt).into() };
        match trainer.step_on_trace(&trace) {
            Ok(loss) => {
                eprintln!("step {i}: loss={loss}");
                if loss > 0.0 {
                    losses.push(loss);
                }
            }
            Err(e) => {
                eprintln!("step {i}: err {e}");
            }
        }
    }
    check("train_steps_executed", trainer.steps >= 1);
    check("train_recent_losses_present", !losses.is_empty());
    if losses.len() >= 4 {
        let first_avg: f32 = losses[..2].iter().copied().sum::<f32>() / 2.0;
        let last_avg: f32 = losses[losses.len() - 2..].iter().copied().sum::<f32>() / 2.0;
        eprintln!("first_avg={first_avg} last_avg={last_avg}");
        check("train_loss_finite", first_avg.is_finite() && last_avg.is_finite());
    }

    let trainable_count = trainer.trainable.all_vars().len();
    check("trainable_var_count_nonzero", trainable_count > 0);
    eprintln!("trainable vars: {trainable_count}");

    let ckpt_dir = std::env::temp_dir().join("sttx-train-witness");
    std::fs::create_dir_all(&ckpt_dir).expect("mk ckpt dir");
    let meta = trainer.meta("random-init-witness".into());
    checkpoint::save(&ckpt_dir, &meta, &trainer.trainable).expect("save");
    let loaded_meta = checkpoint::load_meta(&ckpt_dir).expect("load meta");
    check("checkpoint_meta_round_trip", loaded_meta.steps == trainer.steps);
    check(
        "checkpoint_repo_id",
        loaded_meta.repo_id == "random-init-witness",
    );

    let mut empty_vm = candle_nn::VarMap::new();
    for v in trainer.trainable.all_vars() {
        let _ = v.as_tensor().shape();
    }
    drop(empty_vm.all_vars());
    let _ = &mut empty_vm;

    let f = FAILS.load(Ordering::SeqCst);
    assert_eq!(f, 0, "{f} train-witness assertions failed");
}
