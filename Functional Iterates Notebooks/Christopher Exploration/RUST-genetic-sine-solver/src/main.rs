use cgrustplot::plots::func_plot::function_plot;
use rand::Rng;
use rayon::prelude::*;
const PI: f64 = std::f64::consts::PI;

fn rand_vec(len: usize, range: (f64, f64)) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    
    (0..len)
        .map(|_| rng.gen_range(range.0..range.1))
        .collect()
} 

// Use a reiman sum to integrate the loss of the function
fn loss<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(f: F, g: G) -> f64 {
    const RESOLUTION: f64 = 0.01;
    RESOLUTION * (((-2. * PI / RESOLUTION) as i32)..(2. * PI / RESOLUTION) as i32)
    .map(|i| i as f64 * RESOLUTION)
    .map(|x| (g(x) - f(x)).powi(2))
    .sum::<f64>()
}

// Generate a function from a fourier series
fn fourier(coefs: Vec<f64>) -> impl Fn(f64) -> f64 {
    move |x: f64| coefs.clone()
        .into_iter()
        .enumerate()
        .map(|(i, c)|
            c * (i as f64 * x).sin()
        ).sum::<f64>()
}

#[derive(Debug, Clone)]
struct FourierApprox {
    coefs: Vec<f64>,
    loss: Option<f64>,
}

impl FourierApprox {
    fn from(coefs: Vec<f64>) -> Self {
        Self {
            coefs,
            loss: None,
        }
    }

    fn eval(&mut self) -> &Self {

        let func = fourier(self.coefs.clone());

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

fn eval_all(v: &mut Vec<FourierApprox>) {
    v
    .par_iter_mut()
    .for_each(|ta|
        {ta.eval();}
    );
}

fn sort_approx(v: &mut Vec<FourierApprox>) {
    eval_all(v);

    v.sort_by(|a, b| a.loss.unwrap().partial_cmp(&b.loss.unwrap()).unwrap_or(std::cmp::Ordering::Equal))
}

fn random_approximations(num: usize, range: (f64, f64), coef_number: usize) -> Vec<FourierApprox> {
    (0..num).map(|_|
        FourierApprox::from(rand_vec(coef_number, range))
    )
    .collect()
}

/// Increases the number of fourier terms in a list of approximations
fn elongate(v: Vec<FourierApprox>, n: usize) -> Vec<FourierApprox> {
    v
    .into_iter()
    .map(|ta| {
        let mut new_coefs = ta.coefs;
        new_coefs.extend((0..n).map(|_| 0.));
        FourierApprox::from(new_coefs)
    }).collect()
}

fn avg_loss(v: &Vec<FourierApprox>) -> f64 {
    v.iter().map(|ta| ta.loss.unwrap_or(0.)).sum::<f64>() / v.len() as f64
}

fn genetic_optimize(approxes: &Vec<FourierApprox>, gens: u32, max_num: usize, min_num: usize, temp: f64, decreasing_temp: bool) -> Vec<FourierApprox> {

    let mut v = approxes.clone();
    for i in 0..gens {

        // Adjust temperature value for annealing-like behavior
        let t = if decreasing_temp {temp / (1. + 0.05 * i as f64).sqrt()} else {temp};

        println!("Gen {i:03}/{gens} | temp: {t:.3} | Average Loss: {:.8}", avg_loss(&v));
        
        // Generate mutations
        while v.len() < max_num {
            let extras: Vec<FourierApprox> = v.iter().map(|ta| ta.mutate(t)).collect();
            v.extend(extras);
        }

        // Sort by loss
        sort_approx(&mut v);

        // Select the best
        v = v[..min_num].to_vec();
    }
    
    v
}

fn main() {

    // Original random input
    let mut out: Vec<FourierApprox> = random_approximations(256, (-10., 10.), 16);
    
    // Optimize, then add more terms to the fourier series, then optimize again
    out = genetic_optimize(&out, 100, 256, 64, 10., true);
    out = elongate(out, 16);
    out = genetic_optimize(&out, 1000, 64, 16, 1., true);
    
    // Find best approximation
    sort_approx(&mut out);
    let best = (&out[0]).clone();

    // Format the coefficients of the best approximation to be able to be copied into wolfram
    let formatted_coefs_original: Vec<String> = best.coefs.iter().map(|i| format!("{:.16}, ", i)).collect();
    let mut formatted_coefs = vec![String::from("{")];
    formatted_coefs.extend(formatted_coefs_original);
    let fcl = formatted_coefs.len() - 1;
    formatted_coefs[fcl] = formatted_coefs[fcl][..(formatted_coefs[fcl].len()-2)].to_string();
    formatted_coefs.push(String::from("}"));
    let coefs_string = formatted_coefs.join("");
    println!("Best Coefficients: \n{coefs_string}");

    // Use cgrustplot to plot the aprpoximation in the console
    let func = fourier(best.coefs.clone());
    let nested_func = move |x| func(func(x));
    let func = fourier(best.coefs.clone());
    function_plot(nested_func)
        .print();
    function_plot(func)
        .print();
    
}
