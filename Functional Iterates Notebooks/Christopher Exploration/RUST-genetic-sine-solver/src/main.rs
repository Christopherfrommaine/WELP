use cgrustplot::plots::func_plot::function_plot;
use rand::Rng;
const PI: f64 = std::f64::consts::PI;

fn rand_vec(len: usize, range: (f64, f64)) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    
    (0..len)
        .map(|_| rng.gen_range(range.0..range.1))
        .collect()
} 

fn loss<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(f: F, g: G) -> f64 {
    const RESOLUTION: f64 = 0.005;
    RESOLUTION * (((-2. * PI / RESOLUTION) as i32)..(2. * PI / RESOLUTION) as i32)
    .map(|i| i as f64 * RESOLUTION)
    .map(|x| (g(x) - f(x)).powi(2))
    .sum::<f64>()
}

fn taylor(coefs: Vec<f64>) -> impl Fn(f64) -> f64 {
    move |x: f64| coefs.clone()
        .into_iter()
        .enumerate()
        .map(|(i, c)|
            c * (i as f64 * x).sin()
        ).sum::<f64>()
}

#[derive(Debug, Clone)]
struct TaylorApprox {
    coefs: Vec<f64>,
    loss: Option<f64>,
}

impl TaylorApprox {
    fn from(coefs: Vec<f64>) -> Self {
        Self {
            coefs,
            loss: None,
        }
    }

    fn eval(&mut self) -> &Self {

        let func = taylor(self.coefs.clone());

        self.loss = Some(loss(|x: f64| x.sin(), |x| func(func(x))));
        self

    }

    fn mutate(&self, temp: f64) -> Self {
        let adj: Vec<f64> = rand_vec(self.coefs.len(), (-temp, temp));
        let adj_coef: Vec<f64> = self.coefs
            .iter()
            .zip(
                adj.iter()
            )
            .map(|(a, b)| a + b)
            .collect();
        
        Self::from(adj_coef)
    }
}

fn eval_all(v: &mut Vec<TaylorApprox>) {
    v
    .iter_mut()
    .for_each(|ta|
        {ta.eval();}
    );
}

fn sort_approx(v: &mut Vec<TaylorApprox>) {
    eval_all(v);

    v.sort_by(|a, b| a.loss.unwrap().partial_cmp(&b.loss.unwrap()).unwrap_or(std::cmp::Ordering::Equal))
}

fn random_approximations(num: usize, range: (f64, f64), coef_number: usize) -> Vec<TaylorApprox> {
    (0..num).map(|_|
        TaylorApprox::from(rand_vec(coef_number, range))
    )
    .collect()
}

fn elongate(v: Vec<TaylorApprox>, n: usize) -> Vec<TaylorApprox> {
    v
    .into_iter()
    .map(|ta| {
        let mut new_coefs = ta.coefs;
        new_coefs.extend((0..n).map(|_| 0.));
        TaylorApprox::from(new_coefs)
    }).collect()
}

fn avg_loss(v: &Vec<TaylorApprox>) -> f64 {
    v.iter().map(|ta| ta.loss.unwrap_or(0.)).sum::<f64>() / v.len() as f64
}

fn genetic_optimize(approxes: &Vec<TaylorApprox>, gens: u32, max_num: usize, min_num: usize, temp: f64, decreasing_temp: bool) -> Vec<TaylorApprox> {

    let mut v = approxes.clone();
    for i in 0..gens {

        let t = if decreasing_temp {temp / (1. + 0.05 * i as f64).sqrt()} else {temp};

        println!("Gen {i}/{gens} | Average Loss: {:.8}", avg_loss(&v));

        while v.len() < max_num {
            let extras: Vec<TaylorApprox> = v.iter().map(|ta| ta.mutate(t)).collect();
            v.extend(extras);
        }

        sort_approx(&mut v);

        v = v[..min_num].to_vec();
    }
    
    v
}

fn main() {

    let inp: Vec<TaylorApprox> = random_approximations(256, (-10., 10.), 8);


    let mut out: Vec<TaylorApprox> = genetic_optimize(&inp, 500, 256, 128, 10., true);

    out = elongate(out, 8);
    out = genetic_optimize(&out, 500, 64, 16, 10., true);
    out = elongate(out, 16);
    out = genetic_optimize(&out, 100, 64, 16, 1., true);
    out = genetic_optimize(&out, 1_000, 32, 8, 10., true);
    out = elongate(out, 32);
    out = genetic_optimize(&out, 100, 64, 16, 1., true);
    out = genetic_optimize(&out, 1_000, 32, 8, 10., true);
    out = elongate(out, 64);
    out = genetic_optimize(&out, 100, 64, 16, 1., true);
    out = genetic_optimize(&out, 1_000, 32, 8, 10., true);
    out = elongate(out, 128);
    out = genetic_optimize(&out, 100, 64, 16, 1., true);
    out = genetic_optimize(&out, 5_000, 32, 8, 1., true);


    sort_approx(&mut out);
    let best = (&out[0]).clone();

    println!("Best: {best:?}");

    let formatted_coefs_original: Vec<String> = best.coefs.iter().map(|i|
        format!("{:.16}, ", i)
    ).collect();

    let mut formatted_coefs = vec![String::from("{")];
    formatted_coefs.extend(formatted_coefs_original);
    let fcl = formatted_coefs.len() - 1;
    formatted_coefs[fcl] = formatted_coefs[fcl][..(formatted_coefs[fcl].len()-2)].to_string();
    formatted_coefs.push(String::from("}"));
    let coefs_string = formatted_coefs.join("");

    println!("Best Coefficients: \n{coefs_string}");

    function_plot(taylor(best.coefs.clone()))
        .print();


    
}

