
#import "template.typ": *

#show: project.with(
  title: [Chroma Simulation \ 8Silicon35\_87 \ test],
  authors: ((name: "Pocar Lab", email: "Lab 21", affiliation: "nEXO Collaboration"),),
  date: "2023-07-22",
)

= Experiment Details

- Experiment Name: 8Silicon35\_87
- Number of Particles: 10000
- Random Seed: 1042
- Run ID: 1
- Excluded Geometry: ['reflector1', 'reflector3', 'reflector4', 'reflector5', 'reflector6', 'reflector7', 'reflector8']
- PTE: 0.7601 $plus.minus$ 0.0087.

#pagebreak()
= Material Properties

== Bulk Materials

#table(
  columns: 5,
  [name], [refractive_index], [absorption_length], [scattering_length], [density],
  [liquid xenon], [1.69], [350000000000.0], [35000000000000], [2.942],
  [copper], [0.9733], [0.05], [1000000], [8.96],
  [silicon], [0.682], [1000000.0], [10000000], [2.329],
  [steel], [1.29], [100000000.0], [1000000000], [7.75],
  [teflon], [0.0], [100000000.0], [1000000000], [7.75],
  [silica], [1.644], [100000000.0], [1000000000], [2.202],
  [aluminum], [0.09216], [1000000.0], [10000000], [2.7],
  [killing material], [1.0], [100000000.0], [1000000000], [0.0],
)

#table(
  columns: 6,
  [name], [abs(r_i_error)], [eta], [k], [abs(eta_error)], [abs(k_error)],
  [liquid xenon], [0.0], [0.0], [0.0], [0], [0],
  [copper], [0.0], [0.972], [1.5004], [0], [0],
  [silicon], [0.0], [0.83987], [1.9019], [0], [0],
  [steel], [0.0], [0.0], [0.0], [0], [0],
  [teflon], [0.0], [0.0], [0.0], [0], [0],
  [silica], [0.0], [0.0], [0.0], [0], [0],
  [aluminum], [0.0], [0.09216], [1.9217], [0], [0],
  [killing material], [0.0], [0.0], [0.0], [0], [0],
)

== Surface Properties

#table(
  columns: 6,
  [name], [outer_mat], [inner_mat], [model_id], [reflect_specular], [reflect_diffuse],
  [FBK HD3], [liquid xenon], [silicon], [5], [0], [0],
  [Cu-Xe], [liquid xenon], [copper], [8], [0], [1],
  [silicon-Xe], [liquid xenon], [silicon], [4], [0], [1],
  [killing surface], [None], [None], [8], [0], [0],
)

= Results

== Photon Transmission Efficiency

The Photon Transmission Efficiency (PTE) for this experiment was 
0.7601 $plus.minus$ 0.0087.

== Tallies

#table(
  columns: (auto, auto),
  [Tally], [Count],
  
)
#pagebreak()
= Plots

#let plot_files = ("plots/reflected_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png")


#grid(
  columns: 2,
  gutter: 10pt,
  ..plot_files.map(file => {
    figure(
      image(file, width: 100%),
      caption: file.split("/").at(-1).replace("_", " ").replace(".png", "")
    )
  })
)


